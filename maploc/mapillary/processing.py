# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import logging
from enum import auto, Enum
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from mapillary.vision.common.utils import fileio
from mapillary.vision.sfm.mapillary_sfm.dataset.cluster import ClusterDataSet
from omegaconf import DictConfig  # @manual
from opensfm.context import parallel_map
from opensfm.pymap import Shot
from opensfm.types import Reconstruction
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from ..data.utils import decompose_rotmat
from ..utils.geo import BoundaryBox, Projection
from ..utils.tools import Timer
from .osm_filter import BuildingFilter
from .plane import check_plane_fit, fit_ground_plane
from .reconstruction import (
    filter_reconstruction_points,
    get_undistorters,
    keyframe_selection,
    recover_shot_order,
    render_panorama,
    shots_to_datum,
    undistort_camera,
)
from .triangulation import get_triangulated_reconstruction
from .viz import plot_trajectories_on_map

logger = logging.getLogger(__name__)


class OutputStatus(Enum):
    TOO_FEW_IMAGES = auto()
    LIKELY_MISALIGNED = auto()
    PLANE_FIT_FAILED = auto()
    SUCCESS = auto()


def get_dataset_bbox(dataset, chunk_ids, chunk2data, projection):
    camera_centers = []
    for chunk_id in tqdm(chunk_ids):
        (rec,) = dataset.load_cluster_reconstruction_aligned(
            chunk2data[chunk_id]["sfm_cluster_key"]
        )
        for shot in rec.shots.values():
            camera_centers.append(rec.reference.to_lla(*shot.pose.get_origin())[:2])
    projected = projection.project(np.array(camera_centers))
    return BoundaryBox(projected.min(0), projected.max(0))


def undistort_all_shots(
    dataset: ClusterDataSet,
    undistorters,
    shots: List[Shot],
    pano_offset: float,
    image_dir: Optional[Path] = None,
):
    shots_out = []
    images = []
    for shot_orig in shots:
        undistorter = undistorters[shot_orig.camera.id]
        ret = undistort_shot(shot_orig, dataset, undistorter, pano_offset, image_dir)
        if ret is None:
            continue
        shots, undistorted = ret
        shots_out.extend(shots)
        images.extend(undistorted)
    return shots_out, images


def undistort_all_shots_parallel(
    dataset: ClusterDataSet,
    undistorters,
    shots: List[Shot],
    pano_offset: float,
    num_proc: int,
    image_dir: Optional[Path] = None,
):
    args = []
    for shot_orig in shots:
        undistorter = undistorters[shot_orig.camera.id]
        args.append((shot_orig, dataset, undistorter, pano_offset, image_dir))
    ret = parallel_map(lambda a: undistort_shot(*a), args, num_proc)
    shots_out = [s for r in ret if r is not None for s in r[0]]
    images = [i for r in ret if r is not None for i in r[1]]
    return shots_out, images


def undistort_shot(
    shot_orig: Shot,
    dataset: ClusterDataSet,
    undistorter,
    pano_offset: float,
    image_dir: Optional[Path] = None,
):
    image_raw = dataset.load_image(shot_orig.id)
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
        shots, undistorted = undistort_camera(shot_orig, image_raw, undistorter)
        shots, undistorted = [shots], [undistorted]
    else:
        raise NotImplementedError(camera.projection_type)
    if image_dir is not None:
        for shot, image in zip(shots, undistorted):
            cv2.imwrite(str(image_dir / f"{shot.id}.jpg"), image[..., ::-1])
    return shots, undistorted


def compute_geo_rotation(reference, projection, xyz_orig):
    xy = projection.project(np.array(reference.to_lla(*xyz_orig)[:2]))
    xyz_2 = xyz_orig + np.array([1, 0, 0])
    xy_2 = projection.project(np.array(reference.to_lla(*xyz_2)[:2]))
    diff = xy_2 - xy
    angle = np.rad2deg(np.arctan2(*diff[::-1]))
    R_topo2epsg = Rotation.from_euler("Z", angle, degrees=True).as_matrix()
    return R_topo2epsg


def pack_view(shot, reference, projection, plane, pano_offset):
    xyz_orig = shot.pose.get_origin()
    lla = reference.to_lla(*xyz_orig)
    xy = projection.project(np.array(lla[:2]))
    xyz = np.r_[xy, lla[-1]]
    R_topo2epsg = compute_geo_rotation(reference, projection, xyz_orig)
    R_c2w = R_topo2epsg @ shot.pose.get_R_cam_to_world()
    rpy = decompose_rotmat(R_c2w)
    view = {
        "camera_id": shot.camera.id,
        "R_c2w": R_c2w,
        "roll_pitch_yaw": rpy,
        "t_c2w": xyz,
        "latlong": lla[:2],
    }
    if plane is not None:
        local_plane_params = np.r_[
            R_topo2epsg @ plane.normal,
            plane.d + np.dot(xyz_orig, plane.normal),
        ]
        view["height"] = local_plane_params[-1] / local_plane_params[-2]
        view["plane_params"] = local_plane_params
    if shot.camera.id == "perspective_from_pano":
        view["panorama_offset"] = pano_offset
    for attr_name in [
        "capture_time",
        "gps_position",
        "gps_accuracy",
        "compass_angle",
        "compass_accuracy",
    ]:
        attr = getattr(shot.metadata, attr_name)
        if attr.has_value:
            value = attr.value
            if attr_name == "gps_position":
                value = reference.to_lla(*value)
            view[attr_name] = value
    return view


def process_chunk(
    dataset: ClusterDataSet,
    rec: Reconstruction,
    seed: int,
    projection: Projection,
    building_filter: BuildingFilter,
    image_dir: Path,
    cfg: DictConfig,
):
    ret = None
    if len(rec.shots) < cfg.min_num_images:
        return OutputStatus.TOO_FEW_IMAGES, ret

    shots_ordered = recover_shot_order(rec)
    shots_ordered = np.array(
        [k for k in shots_ordered if k in dataset.thumbnail_key_to_id]
    )
    centers = np.stack([rec.shots[k].pose.get_origin() for k in shots_ordered], 0)

    # Check if some shots overlap with OSM buildings
    centers_xy, _ = shots_to_datum(
        [rec.shots[k] for k in shots_ordered], projection, rec.reference
    )
    in_building = building_filter.in_mask(centers_xy, viz=False)
    if in_building.mean() > cfg.max_ratio_in_building:
        if cfg.verbose:
            print(
                f"Misaligned: {in_building.sum()}/{len(in_building)} ({in_building.mean():.3f})"
            )
        return OutputStatus.LIKELY_MISALIGNED, ret

    # Fit the ground plane
    if cfg.do_plane_fitting:
        plane, xyz_subset = fit_ground_plane(rec, centers)
        if plane is None:
            return OutputStatus.PLANE_FIT_FAILED, ret
        plane_fit_failed, invalid_heights = check_plane_fit(plane, centers)
        if plane_fit_failed:
            if cfg.verbose:
                heights = centers[:, -1] - plane.z(*centers.T[:2])
                logger.info(
                    f"Bad plane fit: inl {plane.inliers.sum()}/{len(plane.inliers)} ({plane.inliers.mean()*100:.1f}%%), heights %s %s",
                    np.median(heights[~invalid_heights]),
                    heights[invalid_heights],
                )
            # return OutputStatus.PLANE_FIT_FAILED, ret
    else:
        plane = None

    # Subsample the valid shots
    shot_idxs_valid = np.where(~in_building)[0]
    shot_idxs = shot_idxs_valid[
        keyframe_selection(
            centers[shot_idxs_valid], min_dist=cfg.min_dist_between_keyframes
        )
    ]
    shots_selected = [rec.shots[k] for k in shots_ordered[shot_idxs]]
    if cfg.do_random_pano_offset:
        pano_offset = np.random.RandomState(seed).uniform(-45, 45)
    else:
        pano_offset = 0

    # Undistort the images
    undistorters = get_undistorters(rec.cameras, cfg.max_image_size)
    with Timer("undistortion" if cfg.verbose else None):
        shots_out, _ = undistort_all_shots_parallel(
            dataset, undistorters, shots_selected, pano_offset, cfg.num_proc, image_dir
        )

    # Assemble the views
    views = {}
    for i, shot in enumerate(shots_out):
        view = pack_view(shot, rec.reference, projection, plane, pano_offset)
        view["index"] = i
        views[shot.id] = view
    camera_dict = {}
    for undistorter in undistorters.values():
        camera = undistorter.camera_perspective
        assert camera.projection_type == "perspective"
        K = camera.get_K_in_pixel_coordinates(camera.width, camera.height)
        camera_dict[camera.id] = {
            "id": camera.id,
            "model": "PINHOLE",
            "width": camera.width,
            "height": camera.height,
            "params": K[[0, 1, 0, 1], [0, 1, 2, 2]],
        }
    ret = {
        "views": views,
        "cameras": camera_dict,
    }

    if cfg.include_points:
        # Filter and index the 3D points
        selected_pids = filter_reconstruction_points(rec)
        points = np.stack([rec.points[i].coordinates for i in selected_pids])
        pid2idx = dict(zip(selected_pids, range(len(selected_pids))))
        # Transform the points to local EPSG
        points_lla = np.stack(rec.reference.to_lla(*points.T), 1)
        ret["points"] = np.concatenate(
            [projection.project(points_lla[:, :2]), points_lla[:, -1:]], 1
        )
        # Filter the points visible in each undistorted shot
        for shot in shots_out:
            # get all points observed by the parent shot
            shot_parent = next(s for s in shots_selected if s.id in shot.id)
            landmarks = shot_parent.get_valid_landmarks()
            observations = np.array(
                [pid2idx[lm.id] for lm in landmarks if lm.id in pid2idx], np.int
            )
            if len(observations) > 0:
                # check which points are visible in the undistorted shot
                xyz_cam = shot.pose.transform_many(points[observations])
                visible = xyz_cam[:, -1] > 0
                uv_norm = shot.camera.project_many(xyz_cam)
                uv = shot.camera.normalized_to_pixel_coordinates_many(uv_norm)
                visible &= np.all(
                    (uv >= 0) & (uv < [shot.camera.width, shot.camera.height]), 1
                )
                observations = observations[visible]
                # sort the points by increasing distance to the camera
                distances = np.linalg.norm(
                    shot.pose.get_origin() - points[observations], axis=1
                )
                observations = observations[np.argsort(distances)]
            views[shot.id] = observations

    if cfg.do_plane_fitting:
        ret["plane"] = (plane.a, plane.b, plane.c, plane.d, plane.inliers.sum())

    if cfg.do_plane_fitting and plane_fit_failed:
        return OutputStatus.PLANE_FIT_FAILED, ret
    else:
        return OutputStatus.SUCCESS, ret


def process_all_chunks(
    dataset: ClusterDataSet,
    chunk_ids: List[int],
    chunk2data: Dict[int, Dict],
    projection: Projection,
    building_filter: BuildingFilter,
    image_dir: Path,
    cfg: DictConfig,
):
    stats = {key.name: 0 for key in OutputStatus}
    stats["count"] = 0
    stats["num_images"] = 0
    to_investigate = []

    outputs = {}
    for chunk_id in tqdm(chunk_ids):
        chunk_key = chunk2data[chunk_id]["sfm_cluster_key"]

        if len(dataset.cluster_images(chunk_key)) < cfg.min_num_images:
            logger.info("Skipping cluster %s because it has too few images.", chunk_key)
            continue

        if cfg.do_retriangulation:
            rec = get_triangulated_reconstruction(dataset, chunk_key)
        else:
            (rec,) = dataset.load_cluster_reconstruction_aligned(chunk_key)
            try:
                tracks = dataset.load_cluster_tracks_manager(chunk_key)
            except FileNotFoundError:
                tracks = None
            else:
                rec.add_correspondences_from_tracks_manager(tracks)

        seed = chunk_id % (2**32 - 1)
        try:
            status, ret = process_chunk(
                dataset,
                rec,
                seed,
                projection,
                building_filter,
                image_dir,
                cfg,
            )
        except Exception as e:
            print(chunk_id, chunk_key)
            raise e

        if ret is not None:
            outputs[chunk_id] = ret
        if status in [OutputStatus.LIKELY_MISALIGNED, OutputStatus.PLANE_FIT_FAILED]:
            to_investigate.append((chunk_id, status.name))

        stats[status.name] += 1
        stats["count"] += 1
        stats["num_images"] += len(rec.shots)

    return outputs, stats, to_investigate


def order_outputs_by_sequence(outputs, chunk2data):
    outputs_per_sequence = {}
    for chunk_id, output in outputs.items():
        seq = chunk2data[chunk_id]["sequence_key"]
        if seq not in outputs_per_sequence:
            outputs_per_sequence[seq] = {
                "views": {},
                "cameras": {},
                "points": {},
                "plane": {},
            }
        for k, v in output["views"].items():
            outputs_per_sequence[seq]["views"][k] = {
                **v,
                "camera_id": f"{chunk_id}/{v['camera_id']}",
                "chunk_id": chunk_id,
                "chunk_key": chunk2data[chunk_id]["sfm_cluster_key"],
            }
        outputs_per_sequence[seq]["cameras"].update(
            {f"{chunk_id}/{k}": v for k, v in output["cameras"].items()}
        )
        outputs_per_sequence[seq]["points"][chunk_id] = output.get("points")
        outputs_per_sequence[seq]["plane"][chunk_id] = output.get("plane")
    return outputs_per_sequence


def plot_coverage(
    path: Path, dataset: ClusterDataSet, chunk_ids: List[int], chunk2data
):
    logger.info("Plotting the coverage...")
    chunk2coords = {}
    for i in tqdm(chunk_ids):
        chunk_key = chunk2data[i]["sfm_cluster_key"]
        (rec,) = dataset.load_cluster_reconstruction_aligned(chunk_key)
        chunk2coords[i] = [
            rec.reference.to_lla(*s.pose.get_origin())[:2] for s in rec.shots.values()
        ]
    m = plot_trajectories_on_map(chunk2coords)
    with fileio.open_file(path, "wb") as fwb:
        m.save(fwb)


def plot_coverage_selection(path, dataset, chunk_ids, chunk2data, outputs):
    logger.info("Plotting the coverage after processing...")
    tag2coords = {"invalid": [], "discard": [], "keep": []}
    for i in tqdm(chunk_ids):
        chunk_key = chunk2data[i]["sfm_cluster_key"]
        (rec,) = dataset.load_cluster_reconstruction_aligned(chunk_key)
        coords = [
            rec.reference.to_lla(*s.pose.get_origin())[:2] for s in rec.shots.values()
        ]
        if i in outputs:
            is_kept = [
                any(v.startswith(s) for v in outputs[i]["views"]) for s in rec.shots
            ]
            tag2coords["keep"].extend([c for c, kept in zip(coords, is_kept) if kept])
            tag2coords["discard"].extend(
                [c for c, kept in zip(coords, is_kept) if not kept]
            )
        else:
            tag2coords["invalid"].extend(coords)
    m = plot_trajectories_on_map(tag2coords)
    with fileio.open_file(path, "wb") as fwb:
        m.save(fwb)
