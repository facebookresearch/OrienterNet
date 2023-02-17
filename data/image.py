# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from typing import Callable, Optional

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from ..utils.geometry import from_homogeneous, to_homogeneous
from ..utils.wrappers import Camera


def rectify_image(
    image: np.ndarray,
    cam: Camera,
    roll: float,
    pitch: Optional[float] = None,
    valid: Optional[torch.Tensor] = None,
):
    h, w = image.shape[-2:]
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
    image,
    size,
    fn: Optional[Callable] = None,
    camera: Optional[Camera] = None,
    valid: np.ndarray = None,
):
    """Resize an image to a fixed size, or according to max or min edge."""
    h, w = image.shape[:2]
    if fn is not None:
        assert isinstance(size, int)
        scale = size / fn(h, w)
        h_new, w_new = int(round(h * scale)), int(round(w * scale))
        # TODO: we should probably recompute the scale like in the second case
        scale = (scale, scale)
    else:
        if isinstance(size, (tuple, list, np.ndarray)):
            w_new, h_new = size
        elif isinstance(size, int):
            w_new = h_new = size
        else:
            raise ValueError(f"Incorrect new size: {size}")
        scale = (w_new / w, h_new / h)
    if min(scale) < 1:
        mode = cv2.INTER_AREA
    else:
        mode = cv2.INTER_LINEAR
    # TODO: maybe use pytorch for resizing
    resized = cv2.resize(image, (w_new, h_new), interpolation=mode)
    ret = [resized, scale]
    if camera is not None:
        if scale != (1, 1):
            camera = camera.scale(scale)
        ret.append(camera)
    if valid is not None:
        valid = (
            cv2.resize(valid * 1, (w_new, h_new), interpolation=cv2.INTER_NEAREST) > 0
        )
        ret.append(valid)
    return ret


def pad_image(image, size, camera: Optional[Camera] = None, valid: np.ndarray = None):
    if isinstance(size, int):
        w_new = h_new = size
    elif isinstance(size, (tuple, list, np.ndarray)):
        w_new, h_new = size
    else:
        raise ValueError(f"Incorrect new size: {size}")
    h, w, *c = image.shape
    assert h <= h_new
    assert w <= w_new
    padded = np.zeros([h_new, w_new] + c, image.dtype)
    padded[:h, :w] = image
    padded_valid = np.zeros([h_new, w_new], np.bool)
    padded_valid[:h, :w] = True if valid is None else valid
    if camera is not None:
        camera = camera.crop((0, 0), (w_new, h_new))
        return padded, padded_valid, camera
    else:
        return padded, padded_valid


def center_pad_crop_image(
    image, size, camera: Optional[Camera] = None, valid: np.ndarray = None
):
    if isinstance(size, int):
        w_new = h_new = size
    elif isinstance(size, (tuple, list, np.ndarray)):
        w_new, h_new = size
    else:
        raise ValueError(f"Incorrect new size: {size}")
    h, w, *c = image.shape
    # amount of cropping
    diff = np.array([w - w_new, h - h_new])
    left, top = left_top = np.round(diff / 2).astype(int)
    right, bottom = diff - left_top
    slice_out = np.s_[: min(h, h_new), : min(w, w_new)]
    slice_in = np.s_[max(top, 0) : h - max(bottom, 0), max(left, 0) : w - max(right, 0)]
    out = np.zeros([h_new, w_new] + c, image.dtype)
    out[slice_out] = image[slice_in]
    out_valid = np.zeros([h_new, w_new], np.bool)
    out_valid[slice_out] = True if valid is None else valid[slice_in]
    if camera is not None:
        camera = camera.crop((max(left, 0), max(top, 0)), (w_new, h_new))
        return out, out_valid, camera
    else:
        return out, out_valid
