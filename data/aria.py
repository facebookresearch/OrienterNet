# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import json
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import gaia  # @manual=fbsource//arvr/python/gaia:gaia
import matplotlib.pyplot as plt
import numpy as np
import pyvrs  # @manual=fbsource//arvr/python/pyvrs:pyvrs
import quaternion
from kapture.io.csv import (
    records_camera_from_file,
    rigs_from_file,
    sensors_from_file,
    trajectories_from_file,
)
from livemaps.mapping.sparse_map import create_map_context_from_filesystem
from omegaconf import OmegaConf  # @manual
from opensfm.transformations import affine_matrix_from_points
from surreal.aria_research_tools.location_lib import write_per_frame_pose
from tqdm import tqdm

from .. import logger
from ..mapillary.plane import PlaneModel
from ..mapillary.processing import decompose_rotmat
from ..mapillary.reconstruction import keyframe_selection
from ..osm.tiling_v2 import TileManager
from ..utils.geo import BoundaryBox, Projection
from ..utils.io import read_image, write_json

GPS_STREAM_ID = "281-1"
FILENAME_GEOALIGN_GPS = "geoalignment.json"
FILENAME_GEOALIGN_MAPLOC = "geoalignment_refined.json"


def query_gaia_ids(tag, exclude="night"):
    creator = gaia.metadata_definition.RecordingQueryMetadataCreator(
        project="Ariane_ResearchMain", tags=[tag]
    )
    query = creator.create()
    ret = gaia.search_recordings(query)
    print([r["tags"] for r in ret if exclude not in r["tags"]])
    ids = [r["id"] for r in ret if exclude not in r["tags"]]
    return ids


def get_gaia_gt_id(gaia_id):
    creator = gaia.metadata_definition.RecordingQueryMetadataCreator(ids=[gaia_id])
    query = creator.create()
    (ret,) = gaia.search_recordings(query)
    (gt_id,) = ret["groundtruth_ids"]
    return gt_id


def get_child_recording_ids(parent_gaia_id: int) -> List[int]:
    (recording,) = gaia.search_recordings({"ids": [parent_gaia_id]})
    child_recording_ids = recording["child_recordings"]
    return child_recording_ids


def get_stream_project_recording(gaia_id, project_owner="Ariane_ResearchScene"):
    child_ids = get_child_recording_ids(gaia_id)
    for child_id in child_ids:
        if gaia.show(child_id)["project_owner"] == project_owner:
            return child_id
    return gaia_id


def check_if_recording_has_gps(gaia_id):
    child_ids = get_child_recording_ids(gaia_id) or [gaia_id]
    child_ids = [f"gaia:{i}" for i in child_ids]
    reader = pyvrs.VRSReader(
        child_ids, auto_read_configuration_records=True, multi_path=True
    )
    reader.filter_by_recordable_ids(GPS_STREAM_ID)
    num_gps = sum(r.record_type == "data" for r in reader)
    has_gps = num_gps > 2
    return has_gps, num_gps


def get_gt_trajectory(path, gaia_id):
    if not path.exists():
        gt_id = get_gaia_gt_id(gaia_id)
        gaia.download_file(
            gt_id, output_dir=path.parent, destination_file_name=path.stem
        )
    with open(path) as fd:
        gt_dict = json.load(fd)
    ts2pose_rig = {}
    for p in gt_dict["trajectory"]["sampled_poses"]:
        ts = p["center_capture_time_us"] * 1000
        compactse3d = p["anchored_pose_type"]["world_anchored_pose"][
            "transform_parent_trajpose"
        ]
        q_xyzw, t_rig2w = np.split(compactse3d, [4])
        q_rig2w = quaternion.from_float_array(np.r_[q_xyzw[-1], q_xyzw[:3]])
        ts2pose_rig[ts] = (q_rig2w, t_rig2w)
    return ts2pose_rig


def interpolate_pose(ts_sparse, ts_dense, ts2pose_dense):
    valid = (ts_sparse >= ts_dense[0]) & (ts_sparse <= ts_dense[-1])
    num_invalid = (~valid).sum()
    if num_invalid > 5:
        logger.warning("%d timestmaps outside the sampling range", num_invalid)
    ts_sparse = ts_sparse[valid]

    idx = np.searchsorted(ts_dense, ts_sparse)
    idx = np.minimum(idx, len(ts_dense) - 1)
    prev = np.maximum(idx - 1, 0)

    ts1 = ts_dense[prev]
    ts2 = ts_dense[idx]

    t1 = np.array([ts2pose_dense[ts][1] for ts in ts1])
    t2 = np.array([ts2pose_dense[ts][1] for ts in ts2])
    w = ((ts_sparse - ts1) / (ts2 - ts1))[:, None]
    t_interp = t1 * (1 - w) + t2 * w

    q1 = [ts2pose_dense[ts][0] for ts in ts1]
    q2 = [ts2pose_dense[ts][0] for ts in ts2]
    q_interp = quaternion.slerp(q1, q2, ts1, ts2, ts_sparse)

    ts2pose_sparse = {ts: (q, t) for ts, q, t in zip(ts_sparse, q_interp, t_interp)}
    return ts2pose_sparse


def estimate_transform_robust(
    all_xy_gt_rel, all_xy_gps, accuracy, thresholds, min_inliers=50
):
    residuals = None
    t_slam2geo = None
    for i in range(len(thresholds) + 1):
        if residuals is None:
            mask = accuracy < 15
        else:
            mask = residuals < thresholds[i - 1]
        if mask.sum() < min_inliers:
            break
        M = affine_matrix_from_points(
            all_xy_gt_rel[mask].T, all_xy_gps[mask].T, shear=False, scale=False
        )
        if t_slam2geo is not None:
            logger.info(
                "Pose updated by %fm, number of inliers: %d (%d%%)",
                np.linalg.norm(t_slam2geo - M[:2, 2]),
                mask.sum(),
                mask.mean() * 100,
            )
        R_slam2geo, t_slam2geo = M[:2, :2], M[:2, 2]
        all_xy_gt_align = t_slam2geo + np.stack(all_xy_gt_rel) @ R_slam2geo.T
        residuals = np.linalg.norm(all_xy_gps - all_xy_gt_align, axis=1)
    return all_xy_gt_align, residuals, (R_slam2geo, t_slam2geo)


def run_slam(gaia_str, output_dir):
    cmd = f"fbsource/fbcode/buck-out/gen/surreal//spaceport/run_mapping.par --input_vrs {gaia_str} --output_dir {output_dir}"
    print(f"Running SLAM with command:\n{cmd}")
    ret = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True
    )
    print(ret.returncode)
    print(ret.stdout.decode("utf-8")[-5000:])
    if not ret.returncode == 0:
        raise ValueError("Command failed, consider running it manually.")


def run_kapture_convert(gaia_id, csv_poses, output_dir):
    cmd = f"fbsource/fbcode/buck-out/gen/surreal/aria_research_tools/run_aria2kapture.par --vrs={gaia_id} --pose_csv_path={csv_poses} --output_path={output_dir}"
    cmd += " --pinhole_rectify --rotate_clockwise_90 --extract_stream_type=RGB"
    cmd += " --rgb_image_width=640 --rgb_image_height=640 --rgb_camera_fx=320 --rgb_camera_fy=320"
    print(f"Running Kapture conversion with command:\n{cmd}")
    ret = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True
    )
    print(ret.returncode)
    print(ret.stdout.decode("utf-8")[-5000:])
    if not ret.returncode == 0:
        raise ValueError("Command failed, consider running it manually.")


def get_gps_from_recording(gaia_id):
    child_ids = get_child_recording_ids(gaia_id) or [gaia_id]
    child_ids = [f"gaia:{i}" for i in child_ids]
    reader = pyvrs.VRSReader(
        child_ids, auto_read_configuration_records=True, multi_path=True
    )
    reader.filter_by_recordable_ids(GPS_STREAM_ID)
    latlonacc = []
    timestamps = []
    for record in reader:
        if record.record_type != "data":
            continue
        data = record.get_state()["metadata_blocks"][0]
        timestamps.append(data["capture_timestamp_ns"])
        latlonacc.append((data["latitude"], data["longitude"], data["accuracy"]))
    timestamps, latlonacc = np.array(timestamps), np.array(latlonacc)
    if len(timestamps) > 0:
        valid = (latlonacc[:, -1] < 1000) & np.any(latlonacc[:, :2] != 0, 1)
        timestamps, latlonacc = timestamps[valid], latlonacc[valid]
    return timestamps, latlonacc


def compute_geoalignment(
    output_path: Path, tmp_dir: Path, tag=None, gaia_ids=None, visualize=False
):
    if gaia_ids is None:
        assert tag is not None
        gaia_ids = query_gaia_ids(tag=tag, exclude=None)

    all_lla_gps = []
    all_xy_gt_rel = []
    all_indices = []
    for idx, gaia_id in enumerate(gaia_ids):
        timestamps_gps, gps_latlonacc = get_gps_from_recording(gaia_id)
        if len(gps_latlonacc) == 0:
            logger.info("No GPS signal for recording %d", gaia_id)
            continue
        valid = gps_latlonacc[:, 2] < 100
        timestamps_gps, gps_latlonacc = timestamps_gps[valid], gps_latlonacc[valid]
        ts2lla_gps = {ts: lla for ts, lla in zip(timestamps_gps, gps_latlonacc)}

        gt_path = tmp_dir / str(gaia_id) / "gt.mdc"
        gt_path.parent.mkdir(exist_ok=True)
        ts2pose_rig = get_gt_trajectory(gt_path, gaia_id)
        timestamps_gt = np.array(sorted(ts2pose_rig))
        ts2pose_gps = interpolate_pose(timestamps_gps, timestamps_gt, ts2pose_rig)
        timestamps_gps_valid = sorted(ts2pose_gps)

        xy_gt_rel = np.stack([ts2pose_gps[ts][1][:2] for ts in timestamps_gps_valid])
        lla_gps_valid = np.stack([ts2lla_gps[ts] for ts in timestamps_gps_valid])
        logger.info("%d, GPS measurements: %d", gaia_id, len(timestamps_gps_valid))

        all_lla_gps.append(lla_gps_valid)
        all_xy_gt_rel.append(xy_gt_rel)
        all_indices.append(np.full(len(timestamps_gps_valid), idx))

    all_lla_gps = np.concatenate(all_lla_gps)
    all_xy_gt_rel = np.concatenate(all_xy_gt_rel)
    all_indices = np.concatenate(all_indices)
    logger.info("Total GPS measurements: %d", len(all_lla_gps))

    projection = Projection.mercator(all_lla_gps[:, :2])
    all_xy_gps = projection.project(all_lla_gps[:, :2])

    thresholds = [5, 2, 1]
    logger.info("Robust fit with thresholds %s", thresholds)
    all_xy_gt_align, residuals, Rt_slam2geo = estimate_transform_robust(
        all_xy_gt_rel, all_xy_gps, all_lla_gps[:, -1], thresholds
    )

    if visualize:
        plt.figure(dpi=100)
        plt.hist(residuals[residuals < 20], bins=30)
        plt.title("Final alignment residuals")
        print((residuals < 2).sum())

        plt.figure(dpi=200)
        plt.scatter(*all_xy_gt_align.T, c="lime", s=1)
        plt.scatter(
            *all_xy_gt_align[residuals < thresholds[-1]].T,
            c="red",
            s=1,
            label=f"residual<{thresholds[-1]}m",
        )
        plt.legend()
        plt.gca().set_aspect("equal")

    output = {
        "projection": projection.epsg,
        "R_gt2geo": Rt_slam2geo[0].tolist(),
        "t_gt2geo": Rt_slam2geo[1].tolist(),
    }
    write_json(output_path, output)

    return (
        Rt_slam2geo,
        projection,
        all_lla_gps,
        all_xy_gps,
        all_xy_gt_align,
        all_indices,
    )


def match_timestamps(ts_rgb, ts_kf, debug: bool = False):
    idx = np.searchsorted(ts_rgb, ts_kf)
    idx = np.minimum(idx, len(ts_rgb) - 1)
    prev = np.maximum(idx - 1, 0)
    idx_kf2rgb = np.where(
        np.abs(ts_rgb[idx] - ts_kf) < np.abs(ts_rgb[prev] - ts_kf), idx, prev
    )
    if debug:
        err = np.abs(ts_kf - ts_rgb[idx_kf2rgb]) / 1e6
        logger.info("Max/median timestamp error: %f/%f", err.max(), np.median(err))
    return idx_kf2rgb


def fit_plane(pcd_w, t_c2w, num_ransac_iters=100, min_height=1, max_height=2.5):
    z = pcd_w[:, -1]
    z_cam = t_c2w[-1]
    maybe_ground = (z <= (z_cam - min_height)) & (z >= (z_cam - max_height))
    xyz_subset = pcd_w[maybe_ground]
    if len(xyz_subset) < PlaneModel.min_num_points:
        return None
    plane = PlaneModel(xyz_subset, thresh=0.1, max_trials=num_ransac_iters)
    return plane


def process_gaia_id(
    gaia_id: str,
    cfg: OmegaConf,
    tmp_dir: Path,
    image_dir: Path,
    Rt_gt2geo: Tuple[np.ndarray],
):
    root = (tmp_dir / str(gaia_id)).absolute()
    root.mkdir(exist_ok=True, parents=True)
    gaia_str = f"gaia:{gaia_id}"

    if not (root / gaia_str).exists():
        run_slam(gaia_str, root)
    csv_poses = root / gaia_str / "per_frame_poses.csv"
    mdc_path = root / gaia_str / "result0000"
    if not csv_poses.exists():
        # Check if the output from the older Map 2.0 instead of the newer UVP
        if (csv_poses.parent / "reconstruction/per_frame_poses.csv").exists():
            csv_poses = csv_poses.parent / "reconstruction/per_frame_poses.csv"
            mdc_path = mdc_path.parent / "reconstruction/result"
        else:
            Path(str(mdc_path) + ".mapping_session_info.mdc").symlink_to(
                root / gaia_str / "result0000.mapping_session.node.mdc"
            )
            write_per_frame_pose(str(root / gaia_str / "result0000"))
    stream_id = get_stream_project_recording(gaia_id)
    kapture_dir = root / "kapture"
    if not kapture_dir.exists():
        run_kapture_convert(stream_id, csv_poses, kapture_dir)
    if (kapture_dir / str(gaia_id)).exists():  # handle the older format
        stream_id = gaia_id
    kapture_path = kapture_dir / f"{stream_id}/sensors"

    sparse_map, _, _, _ = create_map_context_from_filesystem(
        mdc_prefix=str(mdc_path),
        sparse_map=True,
        pose_graph=False,
        image_reader=False,
        descriptor_db=False,
    )
    frame_ids = sparse_map.frame_ids_sorted_by_timestamp()
    ts_to_frame_id = {
        int(sparse_map.get_frame_info(frame_id).name.split(":")[1]): frame_id
        for frame_id in frame_ids
    }
    timestamps_kf = np.array(list(ts_to_frame_id))

    traj = trajectories_from_file(kapture_path / "trajectories.txt")
    sensors_rect = sensors_from_file(kapture_path / "sensors.txt")
    records_camera = records_camera_from_file(kapture_path / "records_camera.txt")
    sensor_id_rgb = f"{stream_id}_RGB"
    timestamps_rgb = np.array(
        sorted([ts for ts, id_ in records_camera.key_pairs() if id_ == sensor_id_rgb])
    )
    rigs = rigs_from_file(kapture_path / "rigs.txt", [sensor_id_rgb])
    T_rgb2rig = rigs[str(stream_id), sensor_id_rgb].inverse()

    # Read GPS
    timestamps_gps, gps_latlonacc = get_gps_from_recording(gaia_id)
    if len(timestamps_gps) == 0:
        raise ValueError(f"Couldn't find any GPS signal in sequence {gaia_id}.")

    # Subsample RGB frames
    t_rgb = np.array(
        [traj[int(t), sensor_id_rgb].inverse().t.squeeze(-1) for t in timestamps_rgb]
    )
    idx_rgb_select = keyframe_selection(t_rgb, min_dist=cfg.min_dist_between_images)
    timestamps_rgb_select = timestamps_rgb[idx_rgb_select]

    # Associate RGB with GPS
    ts_rgb2gps = dict(
        zip(
            timestamps_rgb_select,
            timestamps_gps[match_timestamps(timestamps_gps, timestamps_rgb_select)],
        )
    )
    ts2latlonacc = dict(zip(timestamps_gps, gps_latlonacc))

    # Obtain GT poses for RGB frames
    Ts_rig2gt_rig = get_gt_trajectory(root / "gt.mdc", gaia_id)
    timestamps_gt = np.array(sorted(Ts_rig2gt_rig))
    Ts_rig2gt_rgb = interpolate_pose(timestamps_rgb, timestamps_gt, Ts_rig2gt_rig)
    timestamps_rgb_select = [
        t for t in timestamps_rgb_select if int(t) in Ts_rig2gt_rgb
    ]

    w, h, fx, fy, cx, cy = sensors_rect[sensor_id_rgb].camera_params
    camera_dict = {
        "id": sensor_id_rgb,
        "model": "PINHOLE",
        "width": w,
        "height": h,
        "params": [fx, fy, cx, cy],
    }
    t_kfs = np.stack(
        [
            sparse_map.get_frame_info(ts_to_frame_id[t])
            .T_frame_world.inverse()
            .translation()
            for t in timestamps_kf
        ]
    )
    gps_latlon = np.array(gps_latlonacc)[:, :2]

    views = {}
    for idx in tqdm(range(len(timestamps_rgb_select))):
        ts = timestamps_rgb_select[idx]
        filename = records_camera[int(ts), sensor_id_rgb]
        image_path = kapture_path / "records_data" / filename
        image = read_image(image_path)
        # Skip image if it is too dark
        if np.median(image.max(-1)) < cfg.min_image_intensity:
            continue

        qt_rig2gt = Ts_rig2gt_rgb[int(ts)]
        R_rig2gt = quaternion.as_rotation_matrix(qt_rig2gt[0])
        R_rgb2geo = Rt_gt2geo[0] @ R_rig2gt @ quaternion.as_rotation_matrix(T_rgb2rig.r)
        t_rgb2geo = Rt_gt2geo[1] + Rt_gt2geo[0] @ (
            qt_rig2gt[1] + R_rig2gt @ T_rgb2rig.t.squeeze(-1)
        )
        rpy = decompose_rotmat(R_rgb2geo)
        *latlon, accuracy = ts2latlonacc[ts_rgb2gps[ts]]

        view = {
            "camera_id": camera_dict["id"],
            "R_c2w": R_rgb2geo,
            "roll_pitch_yaw": rpy,
            "t_c2w": t_rgb2geo,
            "latlong": latlon,
            "capture_time": ts,
            "gps_position": latlon,
            "gps_accuracy": accuracy,
        }

        if not cfg.skip_plane_fitting:
            T_w2rgb = traj[int(ts), sensor_id_rgb]
            T_rgb2w = T_w2rgb.inverse()
            t_c2w = T_rgb2w.t.squeeze(-1)
            idx_kf_close = np.linalg.norm(t_kfs - t_c2w, axis=1) < 4
            ts_kf_close = timestamps_kf[idx_kf_close]
            view_ids = [
                i
                for t in ts_kf_close
                for i in sparse_map.get_frame_info(ts_to_frame_id[t]).attached_view_ids
            ]
            track_ids = {
                t for i in view_ids for t in sparse_map.get_view_info(i).tracks
            }
            p3d = np.stack([sparse_map.get_point_in_world(t) for t in track_ids])

            plane = fit_plane(
                p3d,
                t_c2w,
                min_height=cfg.min_camera_height,
                max_height=cfg.max_camera_height,
            )
            num_inls = np.sum(plane.inliers)
            angle = np.rad2deg(np.arccos(np.abs(plane.normal[-1])))
            camera_height = t_c2w[-1] - plane.z(*t_c2w[:2])
            if (
                num_inls < 30
                or angle > 45
                or not (cfg.min_camera_height <= camera_height <= cfg.max_camera_height)
            ):
                logger.info(
                    f"Maybe wrong fit: #inls={num_inls}, angle={angle:.2f}deg, height={camera_height:.2f}m"
                )
                continue
            view["plane_params"] = local_plane_params = np.r_[
                plane.normal,
                plane.d + np.dot(t_c2w, plane.normal),
            ]
            view["height"] = (local_plane_params[-1] / local_plane_params[-2],)

        name = filename.rsplit(".", 1)[0]
        views[name] = view
        shutil.copy(image_path, image_dir)

    logger.info("Retained %d/%d images", len(views), len(timestamps_rgb_select))
    return views, camera_dict, gps_latlon


def main(
    output_dir: Path,
    gaia_ids: List[int],
    geoalignment: Path,
    tmp_dir: Path,
    cfg: Optional[OmegaConf] = None,
):
    gaia_ids_with_gps = []
    for gaia_id in gaia_ids:
        has_gps, num_gps = check_if_recording_has_gps(gaia_id)
        logger.info(
            "%d %s gps (%d timestamps)",
            gaia_id,
            "has" if has_gps else "doesn't have",
            num_gps,
        )
        if has_gps:
            gaia_ids_with_gps.append(gaia_id)
    logger.info(
        "Retained %d/%d Aria sequences that have GPS.",
        len(gaia_ids_with_gps),
        len(gaia_ids),
    )

    with open(geoalignment, "r") as fp:
        geoalignment = json.load(fp)
    projection = Projection(geoalignment["projection"])
    # Convert to a 3D transform
    R_gt2geo = np.eye(3)
    R_gt2geo[:2, :2] = np.array(geoalignment["R_gt2geo"])
    Rt_gt2geo = (R_gt2geo, np.r_[geoalignment["t_gt2geo"], 0])

    all_gps_latlon = []
    output_dump = {}
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    if cfg is None:
        cfg = default_cfg
    for i, gaia_id in enumerate(gaia_ids_with_gps):
        logger.info(f"====== Working on {gaia_id}: {i}/{len(gaia_ids_with_gps)} =====")
        views, camera_dict, gps_latlon = process_gaia_id(
            gaia_id, cfg, tmp_dir, image_dir, Rt_gt2geo
        )
        output_dump[gaia_id] = {
            "views": views,
            "cameras": {camera_dict["id"]: camera_dict},
        }
        all_gps_latlon.extend(gps_latlon)

    write_json(output_dir / "outputs_per_sequence.json", output_dump)

    all_xy = projection.project(np.array(all_gps_latlon))
    bbox_total = BoundaryBox(np.min(all_xy, 0), np.max(all_xy, 0))
    bbox_tiling = bbox_total + cfg.tiling.margin
    tile_manager = TileManager.from_bbox(
        output_dir,
        projection,
        bbox_tiling,
        cfg.tiling.tile_size,
        cfg.tiling.ppm,
    )
    tile_manager.save(output_dir / "tiles.pkl")

    return output_dump, all_xy, tile_manager


datasets = {
    # Seattle
    "reloc_seattle_downtown": {
        "tag": "surreal-aria-reloc-seattle-downtown",
    },
    "reloc_seattle_pike": {
        "tag": "surreal-aria-reloc-seattle-pike",
    },
    "reloc_seattle_westlake": {
        "tag": "surreal-aria-reloc-seattle-westlake",
    },
    # Detroit
    "reloc_detroit_greektown": {
        "tag": "surreal-aria-reloc-detroit-greektown",
    },
    "reloc_detroit_gcp": {
        "tag": "surreal-aria-reloc-detroit-gcp",
    },
    "reloc_detroit_cp": {
        "tag": "surreal-aria-reloc-detroit-cp",
    },
}

default_cfg = OmegaConf.create(
    {
        "min_dist_between_images": 3,
        "skip_plane_fitting": True,
        "min_camera_height": 1,
        "max_camera_height": 2.5,
        "min_image_intensity": 75,
        "tiling": {
            "tile_size": 128,
            "margin": 128,
            "ppm": 2,
        },
    }
)
