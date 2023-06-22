# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import Optional, Tuple

import numpy as np
import torch
from torch.fft import irfftn, rfftn
from torch.nn.functional import grid_sample, log_softmax, pad

from .metrics import angle_error
from .utils import make_grid, rotmat2d


class TemplateSampler(torch.nn.Module):
    def __init__(self, grid_xz_bev, ppm, num_rotations, optimize=True):
        super().__init__()

        Δ = 1 / ppm
        h, w = grid_xz_bev.shape[:2]
        ksize = max(w, h * 2 + 1)
        radius = ksize * Δ
        grid_xy = make_grid(
            radius,
            radius,
            step_x=Δ,
            step_y=Δ,
            orig_y=(Δ - radius) / 2,
            orig_x=(Δ - radius) / 2,
            y_up=True,
        )

        if optimize:
            assert (num_rotations % 4) == 0
            angles = torch.arange(
                0, 90, 90 / (num_rotations // 4), device=grid_xz_bev.device
            )
        else:
            angles = torch.arange(
                0, 360, 360 / num_rotations, device=grid_xz_bev.device
            )
        rotmats = rotmat2d(angles / 180 * np.pi)
        grid_xy_rot = torch.einsum("...nij,...hwj->...nhwi", rotmats, grid_xy)

        grid_ij_rot = (grid_xy_rot - grid_xz_bev[..., :1, :1, :]) * grid_xy.new_tensor(
            [1, -1]
        )
        grid_ij_rot = grid_ij_rot / Δ
        grid_norm = (grid_ij_rot + 0.5) / grid_ij_rot.new_tensor([w, h]) * 2 - 1

        self.optimize = optimize
        self.num_rots = num_rotations
        self.register_buffer("angles", angles, persistent=False)
        self.register_buffer("grid_norm", grid_norm, persistent=False)

    def forward(self, image_bev):
        grid = self.grid_norm
        b, c = image_bev.shape[:2]
        n, h, w = grid.shape[:3]
        grid = grid[None].repeat_interleave(b, 0).reshape(b * n, h, w, 2)
        image = (
            image_bev[:, None]
            .repeat_interleave(n, 1)
            .reshape(b * n, *image_bev.shape[1:])
        )
        kernels = grid_sample(image, grid.to(image.dtype), align_corners=False).reshape(
            b, n, c, h, w
        )

        if self.optimize:  # we have computed only the first quadrant
            kernels_quad234 = [torch.rot90(kernels, -i, (-2, -1)) for i in (1, 2, 3)]
            kernels = torch.cat([kernels] + kernels_quad234, 1)

        return kernels


def conv2d_fft_batchwise(signal, kernel, padding="same", padding_mode="constant"):
    if padding == "same":
        padding = [i // 2 for i in kernel.shape[-2:]]
    padding_signal = [p for p in padding[::-1] for _ in range(2)]
    signal = pad(signal, padding_signal, mode=padding_mode)
    assert signal.size(-1) % 2 == 0

    padding_kernel = [
        pad for i in [1, 2] for pad in [0, signal.size(-i) - kernel.size(-i)]
    ]
    kernel_padded = pad(kernel, padding_kernel)

    signal_fr = rfftn(signal, dim=(-1, -2))
    kernel_fr = rfftn(kernel_padded, dim=(-1, -2))

    kernel_fr.imag *= -1  # flip the kernel
    output_fr = torch.einsum("bc...,bdc...->bd...", signal_fr, kernel_fr)
    output = irfftn(output_fr, dim=(-1, -2))

    crop_slices = [slice(0, output.size(0)), slice(0, output.size(1))] + [
        slice(0, (signal.size(i) - kernel.size(i) + 1)) for i in [-2, -1]
    ]
    output = output[crop_slices].contiguous()

    return output


class SparseMapSampler(torch.nn.Module):
    def __init__(self, num_rotations):
        super().__init__()
        angles = torch.arange(0, 360, 360 / self.conf.num_rotations)
        rotmats = rotmat2d(angles / 180 * np.pi)
        self.num_rotations = num_rotations
        self.register_buffer("rotmats", rotmats, persistent=False)

    def forward(self, image_map, p2d_bev):
        h, w = image_map.shape[-2:]
        locations = make_grid(w, h, device=p2d_bev.device)
        p2d_candidates = torch.einsum(
            "kji,...i,->...kj", self.rotmats.to(p2d_bev), p2d_bev
        )
        p2d_candidates = p2d_candidates[..., None, None, :, :] + locations.unsqueeze(-1)
        # ... x N x W x H x K x 2

        p2d_norm = (p2d_candidates / (image_map.new_tensor([w, h]) - 1)) * 2 - 1
        valid = torch.all((p2d_norm >= -1) & (p2d_norm <= 1), -1)
        value = grid_sample(
            image_map, p2d_norm.flatten(-4, -2), align_corners=True, mode="bilinear"
        )
        value = value.reshape(image_map.shape[:2] + valid.shape[-4])
        return valid, value


def sample_xyr(volume, xy_grid, angle_grid, nearest_for_inf=False):
    # (B, C, H, W, N) to (B, C, H, W, N+1)
    volume_padded = pad(volume, [0, 1, 0, 0, 0, 0], mode="circular")

    size = xy_grid.new_tensor(volume.shape[-3:-1][::-1])
    xy_norm = xy_grid / (size - 1)  # align_corners=True
    angle_norm = (angle_grid / 360) % 1
    grid = torch.concat([angle_norm.unsqueeze(-1), xy_norm], -1)
    grid_norm = grid * 2 - 1

    valid = torch.all((grid_norm >= -1) & (grid_norm <= 1), -1)
    value = grid_sample(volume_padded, grid_norm, align_corners=True, mode="bilinear")

    # if one of the values used for linear interpolation is infinite,
    # we fallback to nearest to avoid propagating inf
    if nearest_for_inf:
        value_nearest = grid_sample(
            volume_padded, grid_norm, align_corners=True, mode="nearest"
        )
        value = torch.where(~torch.isfinite(value) & valid, value_nearest, value)

    return value, valid


def nll_loss_xyr(log_probs, xy, angle):
    log_prob, _ = sample_xyr(
        log_probs.unsqueeze(1), xy[:, None, None, None], angle[:, None, None, None]
    )
    nll = -log_prob.reshape(-1)  # remove C,H,W,N
    return nll


def nll_loss_xyr_smoothed(log_probs, xy, angle, sigma_xy, sigma_r, mask=None):
    *_, nx, ny, nr = log_probs.shape
    grid_x = torch.arange(nx, device=log_probs.device, dtype=torch.float)
    dx = (grid_x - xy[..., None, 0]) / sigma_xy
    grid_y = torch.arange(ny, device=log_probs.device, dtype=torch.float)
    dy = (grid_y - xy[..., None, 1]) / sigma_xy
    dr = (
        torch.arange(0, 360, 360 / nr, device=log_probs.device, dtype=torch.float)
        - angle[..., None]
    ) % 360
    dr = torch.minimum(dr, 360 - dr) / sigma_r
    diff = (
        dx[..., None, :, None] ** 2
        + dy[..., :, None, None] ** 2
        + dr[..., None, None, :] ** 2
    )
    pdf = torch.exp(-diff / 2)
    if mask is not None:
        pdf.masked_fill_(~mask[..., None], 0)
        log_probs = log_probs.masked_fill(~mask[..., None], 0)
    pdf /= pdf.sum((-1, -2, -3), keepdim=True)
    return -torch.sum(pdf * log_probs.to(torch.float), dim=(-1, -2, -3))


def log_softmax_spatial(x, dims=3):
    return log_softmax(x.flatten(-dims), dim=-1).reshape(x.shape)


@torch.jit.script
def argmax_xy(scores: torch.Tensor) -> torch.Tensor:
    indices = scores.flatten(-2).max(-1).indices
    width = scores.shape[-1]
    x = indices % width
    y = torch.div(indices, width, rounding_mode="floor")
    return torch.stack((x, y), -1)


@torch.jit.script
def expectation_xy(prob: torch.Tensor) -> torch.Tensor:
    h, w = prob.shape[-2:]
    grid = make_grid(float(w), float(h), device=prob.device).to(prob)
    return torch.einsum("...hw,hwd->...d", prob, grid)


@torch.jit.script
def expectation_xyr(
    prob: torch.Tensor, covariance: bool = False
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    h, w, num_rotations = prob.shape[-3:]
    x, y = torch.meshgrid(
        [
            torch.arange(w, device=prob.device, dtype=prob.dtype),
            torch.arange(h, device=prob.device, dtype=prob.dtype),
        ],
        indexing="xy",
    )
    grid_xy = torch.stack((x, y), -1)
    xy_mean = torch.einsum("...hwn,hwd->...d", prob, grid_xy)

    angles = torch.arange(0, 1, 1 / num_rotations, device=prob.device, dtype=prob.dtype)
    angles = angles * 2 * np.pi
    grid_cs = torch.stack([torch.cos(angles), torch.sin(angles)], -1)
    cs_mean = torch.einsum("...hwn,nd->...d", prob, grid_cs)
    angle = torch.atan2(cs_mean[..., 1], cs_mean[..., 0])
    angle = (angle * 180 / np.pi) % 360

    if covariance:
        xy_cov = torch.einsum("...hwn,...hwd,...hwk->...dk", prob, grid_xy, grid_xy)
        xy_cov = xy_cov - torch.einsum("...d,...k->...dk", xy_mean, xy_mean)
    else:
        xy_cov = None

    xyr_mean = torch.cat((xy_mean, angle.unsqueeze(-1)), -1)
    return xyr_mean, xy_cov


@torch.jit.script
def argmax_xyr(scores: torch.Tensor) -> torch.Tensor:
    indices = scores.flatten(-3).max(-1).indices
    width, num_rotations = scores.shape[-2:]
    wr = width * num_rotations
    y = torch.div(indices, wr, rounding_mode="floor")
    x = torch.div(indices % wr, num_rotations, rounding_mode="floor")
    angle_index = indices % num_rotations
    angle = angle_index * 360 / num_rotations
    xyr = torch.stack((x, y, angle), -1)
    return xyr


@torch.jit.script
def mask_yaw_prior(
    scores: torch.Tensor, yaw_prior: torch.Tensor, num_rotations: int
) -> torch.Tensor:
    step = 360 / num_rotations
    step_2 = step / 2
    angles = torch.arange(step_2, 360 + step_2, step, device=scores.device)
    yaw_init, yaw_range = yaw_prior.chunk(2, dim=-1)
    rot_mask = angle_error(angles, yaw_init) < yaw_range
    return scores.masked_fill_(~rot_mask[:, None, None], -np.inf)


def fuse_gps(log_prob, uv_gps, ppm, sigma=10, gaussian=False):
    grid = make_grid(*log_prob.shape[-3:-1][::-1]).to(log_prob)
    dist = torch.sum((grid - uv_gps) ** 2, -1)
    sigma_pixel = sigma * ppm
    if gaussian:
        gps_log_prob = -1 / 2 * dist / sigma_pixel**2
    else:
        gps_log_prob = torch.where(dist < sigma_pixel**2, 1, -np.inf)
    log_prob_fused = log_softmax_spatial(log_prob + gps_log_prob.unsqueeze(-1))
    return log_prob_fused
