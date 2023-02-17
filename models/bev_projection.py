# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import numpy as np
import torch
from torch.nn.functional import grid_sample

from ..utils.geometry import from_homogeneous
from .utils import make_grid, rotmat2d


class PolarProjection(torch.nn.Module):
    def __init__(self, z_max, ppm):
        super().__init__()
        self.z_max = z_max
        self.Δ = Δ = 1 / ppm
        z_steps = torch.arange(Δ, z_max + Δ, Δ)
        self.register_buffer("depth_steps", z_steps, persistent=False)

    def ray_to_column_v(self, height, cam, pitch=None):
        # Construct a polar ray in the 2D Y-Z plane
        ray_z = self.depth_steps
        ray_y = height.unsqueeze(-1).expand((-1,) * height.ndim + ray_z.shape)
        ray_z = ray_z.expand(height.shape + (-1,))
        ray_yz = torch.stack([ray_y, ray_z.flip(-1)], -1)

        if pitch is not None:
            R_pitch = rotmat2d(pitch)
            ray_yz = ray_yz @ R_pitch.transpose(-1, -2)

        # Project the ray to an image column
        f, c = cam.f[..., 1].unsqueeze(-1), cam.c[..., 1].unsqueeze(-1)
        ray_v = from_homogeneous(ray_yz).squeeze(-1) * f + c
        return ray_v

    def ray_to_grid_uv(self, ray_v, width):
        ray_v = ray_v.unsqueeze(-1).repeat_interleave(width, -1)
        ray_u = torch.arange(width, device=ray_v.device, dtype=ray_v.dtype)
        ray_u = ray_u.unsqueeze(-2).expand_as(ray_v)
        grid_uv = torch.stack((ray_u, ray_v.to(ray_v)), -1)
        return grid_uv

    def sample_from_image(self, image, grid_uv):
        height, width = image.shape[-2:]
        size = grid_uv.new_tensor([width, height])

        # TODO: it would be more efficient to instead do a 1D interpolation
        # by collapsing W, D into a single dimension.
        grid_uv_norm = (grid_uv + 0.5) / size * 2 - 1
        image_polar = grid_sample(image, grid_uv_norm, align_corners=False)
        valid = torch.all((grid_uv >= 0) & (grid_uv <= (size - 1)), -1)
        return image_polar, valid

    def forward(self, image, height, cam):
        ray_col = self.ray_to_column_v(height, cam)
        grid_uv = self.ray_to_grid_uv(ray_col, image.shape[-1])
        image, valid = self.sample_from_image(image, grid_uv)
        return image, valid, grid_uv


class PolarProjectionPlane(PolarProjection):
    def plane_to_grid_uv(self, plane, cam, width):
        ray_z = self.depth_steps
        ray_u = torch.arange(width, device=ray_z.device, dtype=ray_z.dtype)
        fu, cu = cam.f[..., None, None, 0], cam.c[..., None, None, 0]
        rays_x = ray_z.unsqueeze(-1) * (ray_u.unsqueeze(-2) - cu) / fu
        grid_xz = torch.stack([rays_x, ray_z.unsqueeze(-1).expand_as(rays_x)], -1)
        grid_xz = grid_xz.flip(-3)  # the depth points upwards in the polar image
        plane = plane[..., None, None, :]  # B, 1, 1, 4
        grid_height = (grid_xz * plane[..., :2]).sum(-1) + plane[..., -1]
        grid_height = grid_height / plane[..., -2]
        grid_hz = torch.stack([grid_height, grid_xz[..., 1]], -1)

        fv, cv = cam.f[..., None, None, 1], cam.c[..., None, None, 1]
        grid_v = from_homogeneous(grid_hz).squeeze(-1) * fv + cv
        grid_uv = torch.stack([ray_u.expand_as(grid_v), grid_v], -1)
        return grid_uv

    def forward(self, image, plane, cam):
        grid_uv = self.plane_to_grid_uv(plane, cam, image.shape[-1])
        image, valid = self.sample_from_image(image, grid_uv)
        return image, valid, grid_uv


class PolarProjectionDepth(torch.nn.Module):
    def __init__(self, z_max, ppm, scale_range, z_min=None, log_prob=False):
        super().__init__()
        self.z_max = z_max
        self.Δ = Δ = 1 / ppm
        self.z_min = z_min = Δ if z_min is None else z_min
        self.scale_range = scale_range
        z_steps = torch.arange(z_min, z_max + Δ, Δ)
        self.register_buffer("depth_steps", z_steps, persistent=False)
        self.log_prob = log_prob

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

    def sample_log_depth_scores(self, pixel_log_depths):
        log_depth_steps = torch.log2(self.depth_steps)
        log_min, log_max = np.log2(self.z_min), np.log2(self.z_max)
        log_depth_norm = (log_depth_steps - log_min) / (log_max - log_min)
        log_depth_norm = log_depth_norm * 2 - 1  # in [-1, 1]

        values = pixel_log_depths.flatten(1, 2).unsqueeze(-1)
        indices = log_depth_norm[None, :, None].expand(len(values), -1, -1)
        indices = torch.stack([torch.zeros_like(indices), indices], -1)
        depth_scores = grid_sample(values, indices, align_corners=True)
        depth_scores = depth_scores.reshape(
            pixel_log_depths.shape[:-1] + (len(self.depth_steps),)
        )
        return depth_scores

    def forward(
        self,
        image,
        pixel_scales=None,
        camera=None,
        polar_depths=None,
        polar_log_depths=None,
    ):
        if polar_depths is None:
            if polar_log_depths is not None:
                polar_depths = self.sample_log_depth_scores(polar_log_depths)
            else:
                assert pixel_scales is not None and camera is not None
                polar_depths = self.sample_depth_scores(pixel_scales, camera)

        if self.log_prob:
            depth_prob = torch.softmax(polar_depths, dim=1)
            cell_score = torch.logsumexp(polar_depths, dim=1, keepdim=True)
        else:
            cell_score = polar_depths.sum(1, keepdim=True)  # score per polar cell
            depth_prob = polar_depths / cell_score.clamp(min=1e-4)
        image_polar = torch.einsum("...dhw,...hwz->...dzw", image, depth_prob)

        return image_polar, cell_score.squeeze(1)


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
