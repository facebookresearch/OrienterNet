# Copyright (c) Meta Platforms, Inc. and affiliates.

import numpy as np
from scipy.spatial.transform import Rotation


def crop_map(raster, xy, size, seed=None):
    h, w = raster.shape[-2:]
    state = np.random.RandomState(seed)
    top = state.randint(0, h - size + 1)
    left = state.randint(0, w - size + 1)
    raster = raster[..., top : top + size, left : left + size]
    xy -= np.array([left, top])
    return raster, xy


def random_rot90(raster, xy, heading, seed=None):
    rot = np.random.RandomState(seed).randint(0, 4)
    heading = (heading + rot * np.pi / 2) % (2 * np.pi)
    h, w = raster.shape[-2:]
    if rot == 0:
        xy2 = xy
    elif rot == 2:
        xy2 = np.array([w, h]) - 1 - xy
    elif rot == 1:
        xy2 = np.array([xy[1], w - 1 - xy[0]])
    elif rot == 3:
        xy2 = np.array([h - 1 - xy[1], xy[0]])
    else:
        raise ValueError(rot)
    raster = np.rot90(raster, rot, axes=(-2, -1))
    return raster, xy2, heading


def random_flip(image, raster, xy, heading, seed=None):
    state = np.random.RandomState(seed)
    if state.rand() > 0.5:  # no flip
        return image, raster, xy, heading
    image = image[:, ::-1]
    h, w = raster.shape[-2:]
    if state.rand() > 0.5:  # flip x
        raster = raster[..., :, ::-1]
        xy = np.array([w - 1 - xy[0], xy[1]])
        heading = np.pi - heading
    else:  # flip y
        raster = raster[..., ::-1, :]
        xy = np.array([xy[0], h - 1 - xy[1]])
        heading = -heading
    heading = heading % (2 * np.pi)
    return image, raster, xy, heading


def decompose_rotmat(R_c2w):
    R_cv2xyz = Rotation.from_euler("X", -90, degrees=True)
    rot_w2c = R_cv2xyz * Rotation.from_matrix(R_c2w).inv()
    roll, pitch, yaw = rot_w2c.as_euler("YXZ", degrees=True)
    # rot_w2c_check = R_cv2xyz.inv() * Rotation.from_euler('YXZ', [roll, pitch, yaw], degrees=True)
    # np.testing.assert_allclose(rot_w2c_check.as_matrix(), R_c2w.T, rtol=1e-6, atol=1e-6)
    # R_plane2c = Rotation.from_euler("ZX", [roll, pitch], degrees=True).as_matrix()
    return roll, pitch, yaw
