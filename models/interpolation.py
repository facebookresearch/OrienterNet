# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

# Adapted from PixLoc, Paul-Edouard Sarlin, ETH Zurich
# https://github.com/cvg/pixloc
# Released under the Apache License 2.0

from typing import Tuple

import torch


@torch.jit.script
def interpolate_tensor_bilinear(tensor, pts, return_gradients: bool = False):
    if tensor.dim() == 3:
        assert pts.dim() == 2
        batched = False
        tensor, pts = tensor[None], pts[None]
    else:
        batched = True

    b, c, h, w = tensor.shape
    scale = torch.tensor([w - 1, h - 1]).to(pts)
    pts = (pts / scale) * 2 - 1
    pts = pts.clamp(min=-2, max=2)  # ideally use the mask instead
    interpolated = torch.nn.functional.grid_sample(
        tensor, pts[:, None], mode="bilinear", align_corners=True
    )
    interpolated = interpolated.reshape(b, c, -1).transpose(-1, -2)

    if return_gradients:
        dxdy = torch.tensor([[1, 0], [0, 1]])[:, None].to(pts) / scale * 2
        dx, dy = dxdy.chunk(2, dim=0)
        pts_d = torch.cat([pts - dx, pts + dx, pts - dy, pts + dy], 1)
        tensor_d = torch.nn.functional.grid_sample(
            tensor, pts_d[:, None], mode="bilinear", align_corners=True
        )
        tensor_d = tensor_d.reshape(b, c, -1).transpose(-1, -2)
        tensor_x0, tensor_x1, tensor_y0, tensor_y1 = tensor_d.chunk(4, dim=1)
        gradients = torch.stack(
            [(tensor_x1 - tensor_x0) / 2, (tensor_y1 - tensor_y0) / 2], dim=-1
        )
    else:
        gradients = torch.zeros(b, pts.shape[1], c, 2).to(tensor)

    if not batched:
        interpolated, gradients = interpolated[0], gradients[0]
    return interpolated, gradients


def mask_in_image(pts, image_size: Tuple[int, int], pad: int = 1):
    w, h = image_size
    image_size_ = torch.tensor([w - pad - 1, h - pad - 1]).to(pts)
    return torch.all((pts >= pad) & (pts <= image_size_), -1)


@torch.jit.script
def interpolate_tensor(
    tensor, pts, mode: str = "linear", pad: int = 1, return_gradients: bool = False
):
    """Interpolate a 3D tensor at given 2D locations.
    Args:
        tensor: with shape (C, H, W) or (B, C, H, W).
        pts: points with shape (N, 2) or (B, N, 2)
        mode: interpolation mode, `'linear'` or `'cubic'`
        pad: padding for the returned mask of valid keypoints
        return_gradients: whether to return the first derivative
            of the interpolated values (currentl only in cubic mode).
    Returns:
        tensor: with shape (N, C) or (B, N, C)
        mask: boolean mask, true if pts are in [pad, W-1-pad] x [pad, H-1-pad]
        gradients: (N, C, 2) or (B, N, C, 2), 0-filled if not return_gradients
    """
    h, w = tensor.shape[-2:]
    if mode == "cubic":
        pad += 1  # bicubic needs one more pixel on each side
    mask = mask_in_image(pts, (w, h), pad=pad)
    # Ideally we want to use mask to clamp outlier pts before interpolationm
    # but this line throws some obscure errors about inplace ops.
    # pts = pts.masked_fill(mask.unsqueeze(-1), 0.)

    if mode == "linear":
        interpolated, gradients = interpolate_tensor_bilinear(
            tensor, pts, return_gradients
        )
    else:
        raise NotImplementedError(mode)
    return interpolated, mask, gradients


class Interpolator:
    def __init__(self, mode: str = "linear", pad: int = 1):
        self.mode = mode
        self.pad = pad

    def __call__(
        self, tensor: torch.Tensor, pts: torch.Tensor, return_gradients: bool = False
    ):
        return interpolate_tensor(tensor, pts, self.mode, self.pad, return_gradients)
