import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FeatureFusion(nn.Module):
    def __init__ (self,mode, num_layers,embed_dim):
        super().__init__()
        valid_modes = {"mean", "sum", "weighted", "concat"}

        if mode not in valid_modes:
            raise ValueError(
                f"Unsupported fusion mode: {mode}. "
                f"Expected one of {valid_modes}"
            )
        
        self.mode = mode
        self.num_layers = num_layers

        if mode == "weighted":
            # Khởi tạo bằng 0 -> softmax đồng đều -> tương đương mean
            self.logits = nn.Parameter(torch.zeros(num_layers))

        elif mode == "concat":
            self.proj = nn.Sequential(
                nn.Linear(num_layers * embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
            )
    def forward(self, feat_list):
        if len(feat_list) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} features, "
                f"but received {len(feat_list)}"
            )

        if self.mode == "concat":
            # Mỗi feature: [B, L, C]
            # Kết quả concat: [B, L, num_layers * C]
            return self.proj(torch.cat(feat_list, dim=-1))

        feats = torch.stack(feat_list, dim=1)  # [B, N_layers, L, C]

        if self.mode == "mean":
            return feats.mean(dim=1)

        if self.mode == "sum":
            return feats.sum(dim=1)

        # weighted
        weights = torch.softmax(self.logits, dim=0)
        weights = weights.view(1, self.num_layers, 1, 1)
        return (feats * weights).sum(dim=1)

class INP_Former(nn.Module):
    def __init__(
            self,
            encoder,
            bottleneck,
            aggregation,
            decoder,
            target_layers =[2, 3, 4, 5, 6, 7, 8, 9],
            fuse_layer_encoder =[[0, 1, 2, 3, 4, 5, 6, 7]],
            fuse_layer_decoder =[[0, 1, 2, 3, 4, 5, 6, 7]],
            remove_class_token=False,
            encoder_require_grad_layer=[],
            prototype_token=None,
            inp_fusion_type='mean',
    ) -> None:
        super(INP_Former, self).__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.aggregation = aggregation
        self.decoder = decoder
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self.remove_class_token = remove_class_token
        self.encoder_require_grad_layer = encoder_require_grad_layer
        self.prototype_token = prototype_token[0]
        embed_dim = self.prototype_token.shape[-1]

        self.inp_fusion = FeatureFusion(
            mode=inp_fusion_type,
            num_layers=len(target_layers),
            embed_dim=embed_dim,
        )

        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0


    def gather_loss(self, query, keys):
        self.distribution = 1. - F.cosine_similarity(query.unsqueeze(2), keys.unsqueeze(1), dim=-1)
        self.distance, self.cluster_index = torch.min(self.distribution, dim=2)
        gather_loss = self.distance.mean()
        return gather_loss

    def forward(self, x):
        x = self.encoder.prepare_tokens(x)
        B, L, _ = x.shape
        en_list = []
        for i, blk in enumerate(self.encoder.blocks):
            if i <= self.target_layers[-1]:
                if i in self.encoder_require_grad_layer:
                    x = blk(x)
                else:
                    with torch.no_grad():
                        x = blk(x)
            else:
                continue
            if i in self.target_layers:
                en_list.append(x)
        side = int(math.sqrt(en_list[0].shape[1] - 1 - self.encoder.num_register_tokens))

        if self.remove_class_token:
            en_list = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en_list]

        x = self.inp_fusion(en_list)

        agg_prototype = self.prototype_token
        for i, blk in enumerate(self.aggregation):
            agg_prototype = blk(agg_prototype.unsqueeze(0).repeat((B, 1, 1)), x)
        g_loss = self.gather_loss(x, agg_prototype)

        for i, blk in enumerate(self.bottleneck):
            x = blk(x)

        de_list = []
        for i, blk in enumerate(self.decoder):
            x = blk(x, agg_prototype)
            de_list.append(x)
        de_list = de_list[::-1]

        en = [self.fuse_feature([en_list[idx] for idx in idxs]) for idxs in self.fuse_layer_encoder]
        de = [self.fuse_feature([de_list[idx] for idx in idxs]) for idxs in self.fuse_layer_decoder]

        if not self.remove_class_token:  # class tokens have not been removed above
            en = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en]
            de = [d[:, 1 + self.encoder.num_register_tokens:, :] for d in de]

        en = [e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for d in de]
        return en, de, g_loss

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)