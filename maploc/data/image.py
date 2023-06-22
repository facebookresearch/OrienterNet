# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import Callable, Optional, Union, Sequence

import numpy as np
import torch
import torchvision.transforms.functional as tvf
import collections
from scipy.spatial.transform import Rotation

from ..utils.geometry import from_homogeneous, to_homogeneous
from ..utils.wrappers import Camera


def rectify_image(
    image: torch.Tensor,
    cam: Camera,
    roll: float,
    pitch: Optional[float] = None,
    valid: Optional[torch.Tensor] = None,
):
    *_, h, w = image.shape
    grid = torch.meshgrid(
        [torch.arange(w, device=image.device), torch.arange(h, device=image.device)],
        indexing="xy",
    )
    grid = torch.stack(grid, -1).to(image.dtype)

    if pitch is not None:
        args = ("ZX", (roll, pitch))
    else:
        args = ("Z", roll)
    R = Rotation.from_euler(*args, degrees=True).as_matrix()
    R = torch.from_numpy(R).to(image)

    grid_rect = to_homogeneous(cam.normalize(grid)) @ R.T
    grid_rect = cam.denormalize(from_homogeneous(grid_rect))
    grid_norm = (grid_rect + 0.5) / grid.new_tensor([w, h]) * 2 - 1
    rectified = torch.nn.functional.grid_sample(
        image[None],
        grid_norm[None],
        align_corners=False,
        mode="bilinear",
    ).squeeze(0)
    if valid is None:
        valid = torch.all((grid_norm >= -1) & (grid_norm <= 1), -1)
    else:
        valid = (
            torch.nn.functional.grid_sample(
                valid[None, None].float(),
                grid_norm[None],
                align_corners=False,
                mode="nearest",
            )[0, 0]
            > 0
        )
    return rectified, valid


def resize_image(
    image: torch.Tensor,
    size: Union[int, Sequence, np.ndarray],
    fn: Optional[Callable] = None,
    camera: Optional[Camera] = None,
    valid: np.ndarray = None,
):
    """Resize an image to a fixed size, or according to max or min edge."""
    *_, h, w = image.shape
    if fn is not None:
        assert isinstance(size, int)
        scale = size / fn(h, w)
        h_new, w_new = int(round(h * scale)), int(round(w * scale))
        scale = (scale, scale)
    else:
        if isinstance(size, (collections.abc.Sequence, np.ndarray)):
            w_new, h_new = size
        elif isinstance(size, int):
            w_new = h_new = size
        else:
            raise ValueError(f"Incorrect new size: {size}")
        scale = (w_new / w, h_new / h)
    if (w, h) != (w_new, h_new):
        mode = tvf.InterpolationMode.BILINEAR
        image = tvf.resize(image, (h_new, w_new), interpolation=mode, antialias=True)
        image.clip_(0, 1)
        if camera is not None:
            camera = camera.scale(scale)
        if valid is not None:
            valid = tvf.resize(
                valid.unsqueeze(0),
                (h_new, w_new),
                interpolation=tvf.InterpolationMode.NEAREST,
            ).squeeze(0)
    ret = [image, scale]
    if camera is not None:
        ret.append(camera)
    if valid is not None:
        ret.append(valid)
    return ret


def pad_image(
    image: torch.Tensor,
    size: Union[int, Sequence, np.ndarray],
    camera: Optional[Camera] = None,
    valid: torch.Tensor = None,
    crop_and_center: bool = False,
):
    if isinstance(size, int):
        w_new = h_new = size
    elif isinstance(size, (collections.abc.Sequence, np.ndarray)):
        w_new, h_new = size
    else:
        raise ValueError(f"Incorrect new size: {size}")
    *c, h, w = image.shape
    if crop_and_center:
        diff = np.array([w - w_new, h - h_new])
        left, top = left_top = np.round(diff / 2).astype(int)
        right, bottom = diff - left_top
    else:
        assert h <= h_new
        assert w <= w_new
        top = bottom = left = right = 0
    slice_out = np.s_[..., : min(h, h_new), : min(w, w_new)]
    slice_in = np.s_[
        ..., max(top, 0) : h - max(bottom, 0), max(left, 0) : w - max(right, 0)
    ]
    if (w, h) == (w_new, h_new):
        out = image
    else:
        out = torch.zeros((*c, h_new, w_new), dtype=image.dtype)
        out[slice_out] = image[slice_in]
        if camera is not None:
            camera = camera.crop((max(left, 0), max(top, 0)), (w_new, h_new))
    out_valid = torch.zeros((h_new, w_new), dtype=torch.bool)
    out_valid[slice_out] = True if valid is None else valid[slice_in]
    if camera is not None:
        return out, out_valid, camera
    else:
        return out, out_valid
