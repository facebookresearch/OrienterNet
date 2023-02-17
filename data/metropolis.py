# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import json
import subprocess
from pathlib import Path

import numpy as np
from metropolis import Metropolis  # in fbsource/fbcode/mapillary/research/metropolis
from opensfm.geo import TopocentricConverter
from opensfm.pygeometry import Pose
from pyquaternion import Quaternion
from tqdm import tqdm

from .. import logger
from ..mapillary.plane import PlaneModel
from ..mapillary.processing import compute_geo_rotation, decompose_rotmat
from ..mapillary.run import cfg
from ..osm.tiling_v2 import TileManager
from ..utils.geo import BoundaryBox, Projection
from ..utils.io import write_json

SIDES = {"CAM_FRONT", "CAM_RIGHT", "CAM_BACK", "CAM_LEFT"}


def read_pcd(met, sample_record, pcd_key="MVS"):
    mvs_token = sample_record["data"]["MVS"]
    mvs_record = met.get("sample_data", mvs_token)
    with np.load(met.get_sample_data_path(mvs_token)) as pcd:
        pcd = pcd["points"]
    T_w2pcd = pose_from_sample(mvs_record)
    pcd_w = T_w2pcd.transform_inverse_many(pcd)
    return pcd_w


def fit_plane(pcd_w, t_c2w, num_ransac_iters=100, min_height=1, max_height=3):
    z = pcd_w[:, -1]
    z_cam = t_c2w[-1]
    maybe_ground = (z <= (z_cam - min_height)) & (z >= (z_cam - max_height))
    xyz_subset = pcd_w[maybe_ground]
    if len(xyz_subset) < PlaneModel.min_num_points:
        return None
    plane = PlaneModel(xyz_subset, thresh=0.1, max_trials=num_ransac_iters)
    return plane


def pose_from_sample(met, data_record):
    cs_record = met.get("calibrated_sensor", data_record["calibrated_sensor_token"])
    pose_record = met.get("ego_pose", data_record["ego_pose_token"])
    q_v2w = pose_record["rotation"]
    t_v2w = pose_record["translation"]
    q_c2v = cs_record["rotation"]
    t_c2v = cs_record["translation"]
    T_w2v = Pose(Quaternion(q_v2w).rotation_matrix, t_v2w).inverse()
    T_v2c = Pose(Quaternion(q_c2v).rotation_matrix, t_c2v).inverse()
    T_w2c = Pose.compose(T_v2c, T_w2v)
    return T_w2c


def process_sample(met, sample_record):
    tokens = []
    for side in SIDES:
        cam_token = sample_record["data"].get(side)
        if cam_token is None:
            continue
        cam_record = met.get("sample_data", cam_token)
        if cam_record["ego_pose_token"] == "dummy_ego_pose":
            continue
        tokens.append(cam_token)
    return tokens


def parse_data_record(met, data_record):
    filename = data_record["filename"]
    cs_record = met.get("calibrated_sensor", data_record["calibrated_sensor_token"])

    timestamp = data_record["timestamp"]
    K = np.array(cs_record["camera_intrinsic"])
    w, h = (data_record["width"], data_record["height"])
    camera_dict = {
        "id": cs_record["sensor_token"],
        "model": "PINHOLE",
        "width": w,
        "height": h,
        "params": K[[0, 1, 0, 1], [0, 1, 2, 2]],
    }

    T_w2c = pose_from_sample(met, data_record)
    return filename, camera_dict, timestamp, T_w2c


def main(met, gps_dict, dump_path):
    scene2sd2cams = {}
    for scene in met.scene:
        sd_token = scene["first_sample_token"]
        sd2cams = {}
        while True:
            sample_record = met.get("sample", sd_token)
            ids = process_sample(met, sample_record)
            if len(ids) > 0:
                sd2cams[sd_token] = ids
            if sd_token == scene["last_sample_token"]:
                break
            sd_token = sample_record["next_sample"]
        if len(sd2cams):
            scene2sd2cams[scene["token"]] = sd2cams
    logger.info("Found %d images.", sum(map(len, scene2sd2cams.values())))

    sample_tokens = [(sc, sd) for sc, sds in scene2sd2cams.items() for sd in sds]
    sample2plane = {}
    for scene, sd_token in tqdm(sample_tokens):
        sample_record = met.get("sample", sd_token)
        pcd_w = read_pcd(met, sample_record)
        cam = scene2sd2cams[scene][sd_token][0]
        t_c2w = pose_from_sample(met.get("sample_data", cam)).get_origin()
        plane = fit_plane(pcd_w, t_c2w)
        num_inls = np.sum(plane.inliers)
        angle = np.rad2deg(np.arccos(np.abs(plane.normal[-1])))
        camera_height = t_c2w[-1] - plane.z(*t_c2w[:2])
        if num_inls < 30 or angle > 45 or not (1 <= camera_height <= 3):
            logger.info(
                f"Maybe wrong fit: #inls={num_inls}, angle={angle:.2f}deg, "
                f"height={camera_height:.2f}m"
            )
            plane = None
        sample2plane[sd_token] = plane

    reference = met.geo["reference"]
    reference = TopocentricConverter(
        reference["lat"], reference["lon"], reference["alt"]
    )
    projection = Projection("EPSG:6498")

    all_latlon = []
    outputs = {}
    for scene, sample2cams in scene2sd2cams.items():
        views = {}
        cameras = {}
        for sample_token, cams in sample2cams.items():
            plane = sample2plane[sample_token]
            if plane is None:
                print(f"Skipping sample {sample_token}")
                continue
            gps = (gps_dict[sample_token]["lat"], gps_dict[sample_token]["lon"])
            for cam_token in cams:
                cam_record = met.get("sample_data", cam_token)
                filename, camera_dict, timestamp, T_w2c = parse_data_record(
                    met, cam_record
                )

                xyz_orig = T_w2c.get_origin()
                lla = reference.to_lla(*xyz_orig)
                xy = projection.project(np.array(lla[:2]))
                xyz = np.r_[xy, lla[-1]]
                R_topo2epsg = compute_geo_rotation(reference, projection, xyz_orig)
                R_c2w = R_topo2epsg @ T_w2c.get_R_cam_to_world()
                rpy = decompose_rotmat(R_c2w)
                local_plane_params = np.r_[
                    R_topo2epsg @ np.array((plane.a, plane.b, plane.c)),
                    plane.d + np.dot(xyz_orig, plane.normal),
                ]
                view = {
                    "camera_id": camera_dict["id"],
                    "R_c2w": R_c2w,
                    "roll_pitch_yaw": rpy,
                    "t_c2w": xyz,
                    "latlong": lla[:2],
                    "height": local_plane_params[-1] / local_plane_params[-2],
                    "plane_params": local_plane_params,
                    "capture_time": timestamp,
                    "gps_position": gps,
                }
                name = filename.rsplit(".", 1)[0]
                views[name] = view
                cameras[camera_dict["id"]] = camera_dict
                all_latlon.append(lla[:2])

        outputs[scene] = {"views": views, "cameras": cameras}

    write_json(dump_path / "outputs_per_sequence.json", outputs)

    all_xy = projection.project(np.array(all_latlon))
    bbox_total = BoundaryBox(np.min(all_xy, 0), np.max(all_xy, 0))
    bbox_tiling = bbox_total + cfg.tiling.margin
    tile_manager = TileManager.from_bbox(
        dump_path,
        projection,
        bbox_tiling,
        cfg.tiling.tile_size,
        cfg.tiling.ppm,
    )
    tile_manager.save(dump_path / "tiles.pkl")

    ret = subprocess.run(["ln", "-s", str(met.dataroot), str(dump_path / "images")])
    assert ret.returncode == 0


if __name__ == "__main__":
    with open("./data/gps_dict.json") as fd:
        gps_dict = json.load(fd)
    gps_dict = {v["orig_fname"]: v for v in gps_dict.values()}
    met = Metropolis("train", Path("./buckets/metropolis"))
    dump_path = Path("./data/mapillary_dumps_v2/metropolis_train")
    dump_path.mkdir(exist_ok=True)
    main(met, gps_dict, dump_path)
