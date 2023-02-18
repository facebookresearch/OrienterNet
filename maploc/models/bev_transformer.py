# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import math

import torch
from torch import nn

from .base import BaseModel
from .bev_projection import PolarProjection


class QuadraticAttention(nn.Module):
    def forward(self, q, k, v, source_mask=None):
        dim = q.shape[-1]
        scores = torch.einsum("...nhd,...mhd->...nmh", q, k) / dim**0.5
        if source_mask is not None:
            scores.masked_fill_(~source_mask[..., None, :, None], float("-inf"))
        prob = torch.softmax(scores, dim=-2)
        result = torch.einsum("...nmh,...mhd->...nhd", prob, v)
        return result


class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        assert (dim % num_heads) == 0
        self.dim = dim
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        self.proj_qkv = nn.ModuleList(
            [nn.Linear(dim, dim, bias=True) for _ in range(3)]
        )
        self.merge = nn.Linear(dim, dim, bias=True)
        for layer in list(self.proj_qkv) + [self.merge]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.attention = QuadraticAttention()

    def forward(self, x, source, source_mask=None):
        q, k, v = [
            layer(i).reshape(i.shape[:-1] + (self.num_heads, self.dim_head))
            for layer, i in zip(self.proj_qkv, (x, source, source))
        ]
        aggregated = self.attention(q, k, v, source_mask)
        aggregated = self.merge(aggregated.reshape(x.shape))
        return aggregated


class TransformerBlock(nn.Module):
    def __init__(self, dim, norm, activation=nn.GELU):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2, bias=True),
            activation(),
            nn.Linear(dim * 2, dim, bias=True),
        )
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.xavier_uniform_(self.mlp[2].weight)
        nn.init.normal_(self.mlp[0].bias, std=1e-6)
        nn.init.zeros_(self.mlp[2].bias)
        assert norm in ["pre", "post", None]
        self.norm_mlp_pre = nn.LayerNorm(dim) if norm == "pre" else nn.Identity()
        self.norm_mlp_post = nn.LayerNorm(dim) if norm == "post" else nn.Identity()

    def forward(self, x):
        return x + self.norm_mlp_post(self.mlp(self.norm_mlp_pre(x)))


class EncoderBlock(TransformerBlock):
    def __init__(self, dim, num_heads, norm):
        super().__init__(dim, norm)
        self.self = AttentionBlock(dim, num_heads)
        assert norm in ["pre", "post", None]
        self.norm_pre = nn.LayerNorm(dim) if norm == "pre" else nn.Identity()
        self.norm_post = nn.LayerNorm(dim) if norm == "post" else nn.Identity()

    def forward(self, x, mask=None):
        x = x + self.norm_post(self.self(*[self.norm_pre(x)] * 2, mask))
        return super().forward(x)


class DecoderBlock(TransformerBlock):
    def __init__(self, dim, num_heads, norm):
        super().__init__(dim, norm)
        self.self = AttentionBlock(dim, num_heads)
        self.cross = AttentionBlock(dim, num_heads)
        assert norm in ["pre", "post", None]
        if norm == "pre":
            self.norm_pre1 = nn.LayerNorm(dim)
            self.norm_pre2 = nn.LayerNorm(dim)
        else:
            self.norm_pre1 = self.norm_pre2 = nn.Identity()
        if norm == "post":
            self.norm_post1 = nn.LayerNorm(dim)
            self.norm_post2 = nn.LayerNorm(dim)
        else:
            self.norm_post1 = self.norm_post2 = nn.Identity()

    def forward(self, x, source, mask=None):
        x = x + self.norm_post1(self.self(*[self.norm_pre1(x)] * 2))
        x = x + self.norm_post2(
            self.cross(self.norm_pre2(x), self.norm_pre2(source), mask)
        )
        return super().forward(x)


def fourier_positional_encoding(x, dim, period):
    """Fourier series with log-spaced periods between 1 and T."""
    assert dim % 2 == 0
    steps = torch.linspace(0, 1, dim // 2, dtype=x.dtype, device=x.device)
    factor = 2 * math.pi / (period**steps)
    t = x.unsqueeze(-1) * factor
    enc = torch.cat([torch.cos(t), torch.sin(t)], -1)
    return enc


class PolarTransformer(BaseModel):
    default_conf = {
        "z_max": "???",
        "pixel_per_meter": "???",
        "dim": "???",
        "num_heads": 4,
        "num_blocks_encoder": 2,
        "num_blocks_decoder": 2,
        "norm_location": "pre",
        "max_image_height": 512,  # virtual image for positional encoding
        "max_pe_multiplier": 4,
    }

    def _init(self, conf):
        self.ground_projection = PolarProjection(conf.z_max, conf.pixel_per_meter)
        # Sine embeddings have a stddev (=RMSE) of 1/sqrt(2)
        self.invalid_projection = nn.Parameter(torch.randn(conf.dim) / math.sqrt(2))
        # We scale the encodings such that the initial stddev is 1/sqrt(dim)
        scale_encodings = nn.Parameter(torch.tensor(math.log(math.sqrt(2 / conf.dim))))
        self.register_parameter("scale_encodings", scale_encodings)
        self.proj_encoding_z = nn.Linear(conf.dim, conf.dim, bias=True)
        nn.init.xavier_uniform_(self.proj_encoding_z.weight)

        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(conf.dim, conf.num_heads, conf.norm_location)
                for _ in range(conf.num_blocks_encoder)
            ]
        )
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(conf.dim, conf.num_heads, conf.norm_location)
                for _ in range(conf.num_blocks_decoder)
            ]
        )

        self.final = nn.Linear(self.conf.dim, self.conf.dim, bias=True)
        nn.init.xavier_uniform_(self.final.weight)
        nn.init.zeros_(self.final.bias)

    def _forward(self, data):
        image_features = data["image_features"]
        image_mask = data.get("image_mask")
        height, width = image_features.shape[-2:]
        image_features = image_features.permute(0, 3, 2, 1)  # to B,W,H,D
        pos_image = self.column_encoding(height, image_features.device)
        # TODO: should we also learn the scale of the since encoding?
        pos = pos_image[None, None].to(image_features)
        # print(pos.shape, torch.norm(pos, dim=-1), torch.norm(pos, dim=-1).mean(), torch.mean(pos, dim=-2))
        image_features += pos
        for block in self.encoder_blocks:
            image_features = block(image_features, image_mask)

        pos_polar, valid = self.ray_encoding(
            data["camera_height"], data["camera"], height
        )
        polar_features = pos_polar.unsqueeze(1).repeat_interleave(width, dim=1)
        for block in self.decoder_blocks:
            polar_features = block(polar_features, image_features, image_mask)
        polar_features = self.final(polar_features)

        polar_features = polar_features.permute(0, 3, 2, 1)  # to B,D,H,W
        valid = valid.unsqueeze(-1).repeat_interleave(width, dim=-1)

        return {
            "valid": valid,
            "polar_features": polar_features,
            "polar_encoding": pos_polar,
            "image_encoding": pos_image,
        }

    def column_encoding(self, image_height, device):
        step = self.conf.max_image_height / image_height
        pos = torch.arange(0, self.conf.max_image_height, step, device=device)
        enc = fourier_positional_encoding(
            pos, self.conf.dim, self.conf.max_image_height * self.conf.max_pe_multiplier
        )
        return self.scale_encodings.exp() * enc.to(self.scale_encodings)

    def ray_encoding(self, camera_height, camera, image_height):
        ray_z = self.ground_projection.depth_steps.flip(-1)  # N
        pos_z = fourier_positional_encoding(
            ray_z, self.conf.dim, self.conf.z_max * self.conf.max_pe_multiplier
        )  # N,D

        ray_v = self.ground_projection.ray_to_column_v(camera_height, camera)  # B,N
        valid = (ray_v >= 0) & (ray_v <= (image_height - 1))
        ray_v_norm = ray_v / image_height * self.conf.max_image_height
        pos_v = fourier_positional_encoding(
            ray_v_norm,
            self.conf.dim,
            self.conf.max_image_height * self.conf.max_pe_multiplier,
        )  # B,N,D
        pos_v = torch.where(
            valid.unsqueeze(-1), pos_v, self.invalid_projection[None, None]
        )

        pos = self.proj_encoding_z(pos_z) + pos_v  # B,N,D
        pos = self.scale_encodings.exp() * pos
        return pos, valid  # , pos_z, pos_v, ray_z, ray_v
