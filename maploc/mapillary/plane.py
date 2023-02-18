# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import math
from typing import List, Optional, Set

import numpy as np
import open3d as o3d  # @manual=fbsource//arvr/third-party/open3d:open3d_python
from mapillary.vision.sfm.mapillary_sfm.dataset.cluster import ClusterDataSet
from opensfm.types import Reconstruction


class PlaneModel:
    min_num_points = 3

    def __init__(
        self,
        xyz: np.ndarray,
        thresh: int = 0.5,
        max_trials: int = 10000,
    ):
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
        model, inlier_indices = pcd.segment_plane(
            distance_threshold=thresh,
            ransac_n=self.min_num_points,
            num_iterations=max_trials,
        )
        self.a, self.b, self.c, self.d = model

        inliers = np.zeros(len(xyz), np.bool)
        inliers[inlier_indices] = True
        self.inliers = inliers

    def z(self, x: np.ndarray, y: np.ndarray):
        """Returns elevation given x and y"""
        if math.isclose(self.c, 0.0):
            return np.full(x.shape(), -self.d)
        else:
            return (self.a * x + self.b * y + self.d) / -self.c

    @property
    def normal(self):
        """Unit vector normal to the plane"""
        return np.array([self.a, self.b, self.c])

    def project_points(self, Xi: np.ndarray, Yi: np.ndarray, Zi: np.ndarray):
        """Project point to the ground plane"""
        dist_to_plane = Xi * self.a + Yi * self.b + Zi * self.c + self.d
        Xo = Xi - dist_to_plane * self.a
        Yo = Yi - dist_to_plane * self.b
        Zo = Zi - dist_to_plane * self.c
        return Xo, Yo, Zo


def get_ground_labels(dataset: ClusterDataSet) -> Set[str]:
    ground_labels = []
    for label in dataset.segmentation_labels():
        label = label["name"]
        if (
            "flat" in label
            or "void--ground" in label
            or "marking" in label
            or "manhole" in label
            or "curb" in label
            or "marking" in label
            or "road-side" in label
            or "road-median" in label
            or "curb" in label
            or "ground" in label
        ):
            ground_labels.append(label)
    return set(ground_labels)


def get_ground_points(dataset: ClusterDataSet, rec: Reconstruction):
    ground_labels = get_ground_labels(dataset)
    class_names = [c["name"] for c in dataset.segmentation_labels()]
    p3d2ground = {}
    for key, point in rec.points.items():
        is_ground = False
        for shot, obs_id in point.get_observations().items():
            obs = shot.get_observation(obs_id)
            if class_names[obs.segmentation] in ground_labels:
                is_ground = True
                break
        p3d2ground[key] = is_ground
    return p3d2ground


def fit_ground_plane_semantics(
    rec: Reconstruction,
    dataset: ClusterDataSet,
    shots: Optional[List[str]] = None,
    thresh: float = 0.1,
):
    point2ground = get_ground_points(dataset, rec)
    if shots is None:
        shots = rec.shots.keys()
    points_subset = {p.id for k in shots for p in rec.shots[k].get_valid_landmarks()}
    points_subset = [p for p in points_subset if point2ground[p]]
    if len(points_subset) < PlaneModel.min_num_points:
        return None, None
    xyz_subset = np.stack([rec.points[p].coordinates for p in points_subset])
    plane = PlaneModel(xyz_subset, thresh=0.1)
    return plane, xyz_subset


def fit_ground_plane_semantics_window(
    rec: Reconstruction,
    dataset: ClusterDataSet,
    shots_ordered: List[str],
    idx: int,
    margin: int = 5,
):
    slice_ = slice(max(0, idx - margin), idx + margin + 1)
    shots_subset = shots_ordered[slice_]
    return fit_ground_plane_semantics(rec, dataset, shots_subset)


def fit_ground_plane(
    rec: Reconstruction, poses: np.ndarray, min_height: float = 1, max_height: float = 3
):
    if len(rec.points) < PlaneModel.min_num_points:
        return None, None
    xyz = np.stack([p.coordinates for p in rec.points.values()], 0)
    z = xyz[:, -1]
    z_cam = poses[:, -1]
    maybe_ground = (z <= (z_cam.max() - min_height)) & (z >= (z_cam.min() - max_height))
    xyz_subset = xyz[maybe_ground]
    if len(xyz_subset) < PlaneModel.min_num_points:
        return None, None
    plane = PlaneModel(xyz_subset, thresh=0.1)
    return plane, xyz_subset


def check_plane_fit(
    plane: PlaneModel,
    camera_centers: np.ndarray,
    min_inliers: int = 10,
    max_invalid_ratio: float = 0.2,
    min_height: float = 1,
    max_height: float = 3,
    outlier_margin: int = 3,
):
    heights = camera_centers[:, -1] - plane.z(*camera_centers.T[:2])
    invalid = (heights < min_height) | (heights > max_height)
    invalid |= np.abs(heights - np.mean(heights)) > (np.std(heights) * outlier_margin)

    is_fail = (
        np.sum(plane.inliers) < min_inliers or np.mean(invalid) > max_invalid_ratio
    )
    return is_fail, invalid
