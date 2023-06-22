# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
from torch.nn.functional import grid_sample

from ..utils.geometry import from_homogeneous
from .utils import make_grid


class PolarProjectionDepth(torch.nn.Module):
    def __init__(self, z_max, ppm, scale_range, z_min=None):
        super().__init__()
        self.z_max = z_max
        self.Δ = Δ = 1 / ppm
        self.z_min = z_min = Δ if z_min is None else z_min
        self.scale_range = scale_range
        z_steps = torch.arange(z_min, z_max + Δ, Δ)
        self.register_buffer("depth_steps", z_steps, persistent=False)

    def sample_depth_scores(self, pixel_scales, camera):
        scale_steps = camera.f[..., None, 1] / self.depth_steps.flip(-1)
        log_scale_steps = torch.log2(scale_steps)
        scale_min, scale_max = self.scale_range
        log_scale_norm = (log_scale_steps - scale_min) / (scale_max - scale_min)
        log_scale_norm = log_scale_norm * 2 - 1  # in [-1, 1]

        values = pixel_scales.flatten(1, 2).unsqueeze(-1)
        indices = log_scale_norm.unsqueeze(-1)
        indices = torch.stack([torch.zeros_like(indices), indices], -1)
        depth_scores = grid_sample(values, indices, align_corners=True)
        depth_scores = depth_scores.reshape(
            pixel_scales.shape[:-1] + (len(self.depth_steps),)
        )
        return depth_scores

    def forward(
        self,
        image,
        pixel_scales,
        camera,
        return_total_score=False,
    ):
        depth_scores = self.sample_depth_scores(pixel_scales, camera)
        depth_prob = torch.softmax(depth_scores, dim=1)
        image_polar = torch.einsum("...dhw,...hwz->...dzw", image, depth_prob)
        if return_total_score:
            cell_score = torch.logsumexp(depth_scores, dim=1, keepdim=True)
            return image_polar, cell_score.squeeze(1)
        return image_polar


class CartesianProjection(torch.nn.Module):
    def __init__(self, z_max, x_max, ppm, z_min=None):
        super().__init__()
        self.z_max = z_max
        self.x_max = x_max
        self.Δ = Δ = 1 / ppm
        self.z_min = z_min = Δ if z_min is None else z_min

        grid_xz = make_grid(
            x_max * 2 + Δ, z_max, step_y=Δ, step_x=Δ, orig_y=Δ, orig_x=-x_max, y_up=True
        )
        self.register_buffer("grid_xz", grid_xz, persistent=False)

    def grid_to_polar(self, cam):
        f, c = cam.f[..., 0][..., None, None], cam.c[..., 0][..., None, None]
        u = from_homogeneous(self.grid_xz).squeeze(-1) * f + c
        z_idx = (self.grid_xz[..., 1] - self.z_min) / self.Δ  # convert z value to index
        z_idx = z_idx[None].expand_as(u)
        grid_polar = torch.stack([u, z_idx], -1)
        return grid_polar

    def sample_from_polar(self, image_polar, valid_polar, grid_uz):
        size = grid_uz.new_tensor(image_polar.shape[-2:][::-1])
        grid_uz_norm = (grid_uz + 0.5) / size * 2 - 1
        grid_uz_norm = grid_uz_norm * grid_uz.new_tensor([1, -1])  # y axis is up
        image_bev = grid_sample(image_polar, grid_uz_norm, align_corners=False)

        if valid_polar is None:
            valid = torch.ones_like(image_polar[..., :1, :, :])
        else:
            valid = valid_polar.to(image_polar)[:, None]
        valid = grid_sample(valid, grid_uz_norm, align_corners=False)
        valid = valid.squeeze(1) > (1 - 1e-4)

        return image_bev, valid

    def forward(self, image_polar, valid_polar, cam):
        grid_uz = self.grid_to_polar(cam)
        image, valid = self.sample_from_polar(image_polar, valid_polar, grid_uz)
        return image, valid, grid_uz
