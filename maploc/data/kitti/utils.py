# Copyright (c) Meta Platforms, Inc. and affiliates.

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ...utils.geo import Projection

split_files = ["test1_files.txt", "test2_files.txt", "train_files.txt"]


def parse_gps_file(path, projection: Projection = None):
    with open(path, "r") as fid:
        lat, lon, _, roll, pitch, yaw, *_ = map(float, fid.read().split())
    latlon = np.array([lat, lon])
    R_world_gps = Rotation.from_euler("ZYX", [yaw, pitch, roll]).as_matrix()
    t_world_gps = None if projection is None else np.r_[projection.project(latlon), 0]
    return latlon, R_world_gps, t_world_gps


def parse_split_file(path: Path):
    with open(path, "r") as fid:
        info = fid.read()
    names = []
    shifts = []
    for line in info.split("\n"):
        if not line:
            continue
        name, *shift = line.split()
        names.append(tuple(name.split("/")))
        if len(shift) > 0:
            assert len(shift) == 3
            shifts.append(np.array(shift, float))
    shifts = None if len(shifts) == 0 else np.stack(shifts)
    return names, shifts


def parse_calibration_file(path):
    calib = {}
    with open(path, "r") as fid:
        for line in fid.read().split("\n"):
            if not line:
                continue
            key, *data = line.split(" ")
            key = key.rstrip(":")
            if key.startswith("R"):
                data = np.array(data, float).reshape(3, 3)
            elif key.startswith("T"):
                data = np.array(data, float).reshape(3)
            elif key.startswith("P"):
                data = np.array(data, float).reshape(3, 4)
            calib[key] = data
    return calib


def get_camera_calibration(calib_dir, cam_index: int):
    calib_path = calib_dir / "calib_cam_to_cam.txt"
    calib_cam = parse_calibration_file(calib_path)
    P = calib_cam[f"P_rect_{cam_index:02}"]
    K = P[:3, :3]
    size = np.array(calib_cam[f"S_rect_{cam_index:02}"], float).astype(int)
    camera = {
        "model": "PINHOLE",
        "width": size[0],
        "height": size[1],
        "params": K[[0, 1, 0, 1], [0, 1, 2, 2]],
    }

    t_cam_cam0 = P[:3, 3] / K[[0, 1, 2], [0, 1, 2]]
    R_rect_cam0 = calib_cam["R_rect_00"]

    calib_gps_velo = parse_calibration_file(calib_dir / "calib_imu_to_velo.txt")
    calib_velo_cam0 = parse_calibration_file(calib_dir / "calib_velo_to_cam.txt")
    R_cam0_gps = calib_velo_cam0["R"] @ calib_gps_velo["R"]
    t_cam0_gps = calib_velo_cam0["R"] @ calib_gps_velo["T"] + calib_velo_cam0["T"]
    R_cam_gps = R_rect_cam0 @ R_cam0_gps
    t_cam_gps = t_cam_cam0 + R_rect_cam0 @ t_cam0_gps
    return camera, R_cam_gps, t_cam_gps
