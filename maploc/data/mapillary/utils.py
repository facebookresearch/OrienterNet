# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
from typing import List, Tuple

import cv2
import numpy as np
from opensfm import features
from opensfm.pygeometry import Camera, compute_camera_mapping, Pose
from opensfm.pymap import Shot
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)


def keyframe_selection(shots: List[Shot], min_dist: float = 4) -> List[int]:
    camera_centers = np.stack([shot.pose.get_origin() for shot in shots], 0)
    distances = np.linalg.norm(np.diff(camera_centers, axis=0), axis=1)
    selected = [0]
    cum = 0
    for i in range(1, len(camera_centers)):
        cum += distances[i - 1]
        if cum >= min_dist:
            selected.append(i)
            cum = 0
    return selected


def perspective_camera_from_pano(camera: Camera, size: int) -> Camera:
    camera_new = Camera.create_perspective(0.5, 0, 0)
    camera_new.height = camera_new.width = size
    camera_new.id = "perspective_from_pano"
    return camera_new


def scale_camera(camera: Camera, max_size: int) -> Camera:
    height = camera.height
    width = camera.width
    factor = max_size / float(max(height, width))
    if factor >= 1:
        return camera
    camera.width = int(round(width * factor))
    camera.height = int(round(height * factor))
    return camera


class PanoramaUndistorter:
    def __init__(self, camera_pano: Camera, camera_new: Camera):
        w, h = camera_new.width, camera_new.height
        self.shape = (h, w)

        dst_y, dst_x = np.indices(self.shape).astype(np.float32)
        dst_pixels_denormalized = np.column_stack([dst_x.ravel(), dst_y.ravel()])
        dst_pixels = features.normalized_image_coordinates(
            dst_pixels_denormalized, w, h
        )
        self.dst_bearings = camera_new.pixel_bearing_many(dst_pixels)

        self.camera_pano = camera_pano
        self.camera_perspective = camera_new

    def __call__(
        self, image: np.ndarray, panoshot: Shot, perspectiveshot: Shot
    ) -> np.ndarray:
        # Rotate to panorama reference frame
        rotation = np.dot(
            panoshot.pose.get_rotation_matrix(),
            perspectiveshot.pose.get_rotation_matrix().T,
        )
        rotated_bearings = np.dot(self.dst_bearings, rotation.T)

        # Project to panorama pixels
        src_pixels = panoshot.camera.project_many(rotated_bearings)
        src_pixels_denormalized = features.denormalized_image_coordinates(
            src_pixels, image.shape[1], image.shape[0]
        )
        src_pixels_denormalized.shape = self.shape + (2,)

        # Sample color
        x = src_pixels_denormalized[..., 0].astype(np.float32)
        y = src_pixels_denormalized[..., 1].astype(np.float32)
        colors = cv2.remap(image, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        return colors


class CameraUndistorter:
    def __init__(self, camera_distorted: Camera, camera_new: Camera):
        self.maps = compute_camera_mapping(
            camera_distorted,
            camera_new,
            camera_distorted.width,
            camera_distorted.height,
        )
        self.camera_perspective = camera_new
        self.camera_distorted = camera_distorted

    def __call__(self, image: np.ndarray) -> np.ndarray:
        assert image.shape[:2] == (
            self.camera_distorted.height,
            self.camera_distorted.width,
        )
        undistorted = cv2.remap(image, *self.maps, cv2.INTER_LINEAR)
        resized = cv2.resize(
            undistorted,
            (self.camera_perspective.width, self.camera_perspective.height),
            interpolation=cv2.INTER_AREA,
        )
        return resized


def render_panorama(
    shot: Shot,
    pano: np.ndarray,
    undistorter: PanoramaUndistorter,
    offset: float = 0.0,
) -> Tuple[List[Shot], List[np.ndarray]]:
    yaws = [0, 90, 180, 270]
    suffixes = ["front", "left", "back", "right"]
    images = []
    shots = []

    # To reduce aliasing, since cv2.remap does not support area samplimg,
    # we first resize with anti-aliasing.
    h, w = undistorter.shape
    h, w = (w * 2, w * 4)  # assuming 90deg FOV
    pano_resized = cv2.resize(pano, (w, h), interpolation=cv2.INTER_AREA)

    for yaw, suffix in zip(yaws, suffixes):
        R_pano2persp = Rotation.from_euler("Y", yaw + offset, degrees=True).as_matrix()
        name = f"{shot.id}_{suffix}"
        shot_new = Shot(
            name,
            undistorter.camera_perspective,
            Pose.compose(Pose(R_pano2persp), shot.pose),
        )
        shot_new.metadata = shot.metadata
        perspective = undistorter(pano_resized, shot, shot_new)
        images.append(perspective)
        shots.append(shot_new)
    return shots, images


def undistort_camera(
    shot: Shot, image: np.ndarray, undistorter: CameraUndistorter
) -> Tuple[Shot, np.ndarray]:
    name = f"{shot.id}_undistorted"
    shot_out = Shot(name, undistorter.camera_perspective, shot.pose)
    shot_out.metadata = shot.metadata
    undistorted = undistorter(image)
    return shot_out, undistorted


def undistort_shot(
    image_raw: np.ndarray,
    shot_orig: Shot,
    undistorter,
    pano_offset: float,
) -> Tuple[List[Shot], List[np.ndarray]]:
    camera = shot_orig.camera
    if image_raw.shape[:2] != (camera.height, camera.width):
        raise ValueError(
            shot_orig.id, image_raw.shape[:2], (camera.height, camera.width)
        )
    if camera.is_panorama(camera.projection_type):
        shots, undistorted = render_panorama(
            shot_orig, image_raw, undistorter, offset=pano_offset
        )
    elif camera.projection_type in ("perspective", "fisheye"):
        shot, undistorted = undistort_camera(shot_orig, image_raw, undistorter)
        shots, undistorted = [shot], [undistorted]
    else:
        raise NotImplementedError(camera.projection_type)
    return shots, undistorted
