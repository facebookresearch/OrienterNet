# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import logging
import os.path as osp
from typing import Dict, List, Optional, Set, Tuple

import cv2
import fsspec
import numpy as np
import stl.lightning.io.filesystem as stlfs
from mapillary.vision.common.config.buckets import THUMBS_BUCKET
from mapillary.vision.sfm.mapillary_sfm.dataset import ClusterVRSDataset
from opensfm import features
from opensfm.geo import TopocentricConverter
from opensfm.io import imread_from_fileobject
from opensfm.pygeometry import Camera, compute_camera_mapping, Pose
from opensfm.pymap import Shot
from opensfm.types import Reconstruction
from opensfm.undistort import (
    perspective_camera_from_fisheye,
    perspective_camera_from_perspective,
)
from scipy.spatial.transform import Rotation

from ..utils.geo import Projection

logger = logging.getLogger(__name__)


class ClusterVRSDatasetWithThumbnails(ClusterVRSDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_files = {
            i: osp.join(self.data_path, "images", i) for i in self.image_list
        }

    def load_image_list_and_exif(self, all_images: Optional[Set[str]]) -> None:
        self.cluster_ids: Set[str] = set()
        self._init_cluster_ids_list()
        for cluster_id in self.cluster_ids:
            if self.cluster_reconstruction_initial_exists(cluster_id):
                cluster = self.load_cluster_reconstruction_initial(cluster_id)[0]
            elif self.cluster_reconstruction_aligned_exists(cluster_id):
                cluster = self.load_cluster_reconstruction_aligned(cluster_id)[0]
            else:
                logger.warning(
                    "Neither initial or aligned reconstruction for cluster %s",
                    cluster_id,
                )
                continue
            for image_key, shot in cluster.shots.items():
                # Load all images, even those that are not in VRS format
                # if all_images is not None and image_key not in all_images:
                #     continue
                self.exifs[image_key] = self._convert_shot_to_exif(
                    cluster.reference, shot, shot.camera
                )
                self.image_list.append(image_key)

    def load_image(
        self,
        image: str,
        unchanged: bool = False,
        anydepth: bool = False,
        grayscale: bool = False,
    ) -> np.ndarray:
        path = self._image_file(image)
        if not osp.exists(path):
            path = None
            thumbnail_id = self.thumbnail_key_to_id.get(image)
            if thumbnail_id is not None:
                remote_root = f"manifold://{THUMBS_BUCKET}"
                path = f"{remote_root}/flat/{thumbnail_id}"
                if not stlfs.get_filesystem(remote_root).exists(path):
                    path = None
            if path is None:
                raise FileNotFoundError(
                    f"Image {image} does not exist (thumbnail ID: {thumbnail_id})"
                )
        with fsspec.open(path, "rb") as fp:
            return imread_from_fileobject(fp, grayscale, unchanged, anydepth)

    def load_segmentation(self, image: str):
        return None


def recover_shot_order(rec: Reconstruction) -> List[str]:
    shot2time = {
        key: shot.metadata.capture_time.value for key, shot in rec.shots.items()
    }
    shots = np.array(rec.shots)
    shots_ordered = shots[np.argsort([shot2time[k] for k in shots])]
    return shots_ordered


def keyframe_selection(camera_centers: np.ndarray, min_dist: float = 4) -> List[int]:
    distances = np.linalg.norm(np.diff(camera_centers, axis=0), axis=1)
    selected = [0]
    cum = 0
    for i in range(1, len(camera_centers)):
        cum += distances[i - 1]
        if cum >= min_dist:
            selected.append(i)
            cum = 0
    return selected


def shots_to_datum(
    shots: List[Shot], proj: Projection, reference: TopocentricConverter
):
    geo = [reference.to_lla(*s.pose.get_origin())[:2] for s in shots]
    xy = [proj.project(np.array(ll)) for ll in geo]
    return np.stack(xy), geo


def filter_reconstruction_points(
    rec: Reconstruction, max_point_height: int = 100, max_cam_height: int = 3
) -> List[str]:
    """Discard points that are below the ground level or unrealistically high."""
    shot2z = {k: v.pose.get_origin()[-1] for k, v in rec.shots.items()}
    pids = list(rec.points)
    pid2minz = {
        i: min(shot2z[s.id] for s in rec.points[i].get_observations()) for i in pids
    }
    z_min = min(shot2z.values())
    z_max = max(shot2z.values())
    margin = max_cam_height + z_max - z_min
    too_low = np.array(
        [rec.points[i].coordinates[-1] < (pid2minz[i] - margin) for i in pids]
    )
    too_high = np.array(
        [rec.points[i].coordinates[-1] > (z_max + max_point_height) for i in pids]
    )
    valid = ~(too_low | too_high)
    selected = np.array(pids)[valid].tolist()
    logger.info("Keeping %d/%d 3D points.", len(selected), len(pids))
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


def get_undistorters(cameras: Dict[str, Camera], max_size: int) -> Dict:
    undistorters = {}
    for camera in cameras.values():
        projection = camera.projection_type
        if Camera.is_panorama(projection):
            camera_new = perspective_camera_from_pano(camera, max_size)
            undistorter = PanoramaUndistorter(camera, camera_new)
        elif projection in ["fisheye", "perspective"]:
            if projection == "fisheye":
                camera_new = perspective_camera_from_fisheye(camera)
            else:
                camera_new = perspective_camera_from_perspective(camera)
            camera_new = scale_camera(camera_new, max_size)
            camera_new.id = camera.id + "_undistorted"
            undistorter = CameraUndistorter(camera, camera_new)
        else:
            raise NotImplementedError(camera.projection_type)
        undistorters[camera.id] = undistorter
    return undistorters
