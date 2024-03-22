# Copyright (c) Meta Platforms, Inc. and affiliates.

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from maploc.utils.wrappers import Transform2D


def crop_map(raster, xy, size, seed=None):
    h, w = raster.shape[-2:]
    state = np.random.RandomState(seed)
    top = state.randint(0, h - size + 1)
    left = state.randint(0, w - size + 1)
    raster = raster[..., top : top + size, left : left + size]
    xy -= np.array([left, top])
    return raster, xy


def random_rot90(
    raster: torch.Tensor,
    tile_T_cam: Transform2D,
    pixels_per_meter: float,
    seed: int = None,
):
    rot = np.random.RandomState(seed).randint(0, 4)
    raster = torch.rot90(raster, rot, dims=(-2, -1))

    # Rotate the camera position around tile's center
    map_t_center = np.array(raster.shape[-2:]) / 2.0
    tile_t_center = Transform2D.from_pixels(map_t_center, pixels_per_meter)
    center_t_cam = tile_T_cam.t - tile_t_center
    R = Transform2D.from_degrees(torch.Tensor([rot * 90]), torch.zeros(2)).float()
    center_t_rotcam = R @ center_t_cam.T.float()
    tile_t_rotcam = center_t_rotcam.squeeze(0) + tile_t_center
    tile_r_rotcam = (tile_T_cam.angle + rot * 90) % 360
    tile_T_rotcam = Transform2D.from_degrees(tile_r_rotcam, tile_t_rotcam)
    return raster, tile_T_rotcam


def random_flip(
    image: torch.Tensor,
    valid: torch.Tensor,
    raster: torch.Tensor,
    tile_T_cam: Transform2D,
    pixels_per_meter: float,
    seed: int = None,
):
    state = np.random.RandomState(seed)
    if state.rand() > 0.5:  # no flip
        return image, valid, raster, tile_T_cam

    image = torch.flip(image, (-1,))
    valid = torch.flip(valid, (-1,))

    map_t_center = np.array(raster.shape[-2:]) / 2.0
    tile_t_center = Transform2D.from_pixels(map_t_center, pixels_per_meter)
    center_t_cam = tile_T_cam.t - tile_t_center
    if state.rand() > 0.5:  # flip x
        raster = torch.flip(raster, (-1,))
        tile_r_flipcam = 180 - tile_T_cam.angle
        center_t_flipcam = center_t_cam * torch.tensor([-1, 1])
    else:  # flip y
        tile_r_flipcam = -tile_T_cam.angle
        raster = torch.flip(raster, (-2,))
        center_t_flipcam = center_t_cam * torch.tensor([1, -1])
    tile_t_flipcam = center_t_flipcam + tile_t_center
    tile_T_flipcam = Transform2D.from_degrees(tile_r_flipcam % 360, tile_t_flipcam)
    return image, valid, raster, tile_T_flipcam


def decompose_rotmat(R_c2w):
    R_cv2xyz = Rotation.from_euler("X", -90, degrees=True)
    rot_w2c = R_cv2xyz * Rotation.from_matrix(R_c2w).inv()
    roll, pitch, yaw = rot_w2c.as_euler("YXZ", degrees=True)
    # R_plane2c = Rotation.from_euler("ZX", [roll, pitch], degrees=True).as_matrix()
    return roll, pitch, yaw


def compose_rotmat(roll, pitch, yaw):
    rot_w2c = Rotation.from_euler("YXZ", angles=[roll, pitch, yaw], degrees=True)
    R_xyz2cv = Rotation.from_euler("X", 90, degrees=True)
    R_w2c = R_xyz2cv * rot_w2c
    return R_w2c.inv().as_matrix()
