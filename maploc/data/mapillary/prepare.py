# Copyright (c) Meta Platforms, Inc. and affiliates.

import asyncio
import argparse
from collections import defaultdict
import json
import shutil
from pathlib import Path
from typing import List

import numpy as np
import cv2
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
from omegaconf import DictConfig, OmegaConf
from opensfm.pygeometry import Camera
from opensfm.pymap import Shot
from opensfm.undistort import (
    perspective_camera_from_fisheye,
    perspective_camera_from_perspective,
)

from ... import logger
from ...osm.tiling import TileManager
from ...osm.viz import GeoPlotter
from ...utils.geo import BoundaryBox, Projection
from ...utils.io import write_json, download_file, DATA_URL
from ..utils import decompose_rotmat
from .utils import (
    keyframe_selection,
    perspective_camera_from_pano,
    scale_camera,
    CameraUndistorter,
    PanoramaUndistorter,
    undistort_shot,
)
from .download import (
    MapillaryDownloader,
    opensfm_shot_from_info,
    image_filename,
    fetch_image_infos,
    fetch_images_pixels,
)
from .dataset import MapillaryDataModule


location_to_params = {
    "sanfrancisco_soma": {
        "bbox": BoundaryBox(
            [-122.410307, 37.770364][::-1], [-122.388772, 37.795545][::-1]
        ),
        "camera_models": ["GoPro Max"],
        "osm_file": "sanfrancisco.osm",
    },
    "sanfrancisco_hayes": {
        "bbox": BoundaryBox(
            [-122.438415, 37.768634][::-1], [-122.410605, 37.783894][::-1]
        ),
        "camera_models": ["GoPro Max"],
        "osm_file": "sanfrancisco.osm",
    },
    "amsterdam": {
        "bbox": BoundaryBox([4.845284, 52.340679][::-1], [4.926147, 52.386299][::-1]),
        "camera_models": ["GoPro Max"],
        "osm_file": "amsterdam.osm",
    },
    "lemans": {
        "bbox": BoundaryBox([0.185752, 47.995125][::-1], [0.224088, 48.014209][::-1]),
        "owners": ["xXOocM1jUB4jaaeukKkmgw"],  # sogefi
        "osm_file": "lemans.osm",
    },
    "berlin": {
        "bbox": BoundaryBox([13.416271, 52.459656][::-1], [13.469829, 52.499195][::-1]),
        "owners": ["LT3ajUxH6qsosamrOHIrFw"],  # supaplex030
        "osm_file": "berlin.osm",
    },
    "montrouge": {
        "bbox": BoundaryBox([2.298958, 48.80874][::-1], [2.332989, 48.825276][::-1]),
        "owners": [
            "XtzGKZX2_VIJRoiJ8IWRNQ",
            "C4ENdWpJdFNf8CvnQd7NrQ",
            "e_ZBE6mFd7CYNjRSpLl-Lg",
        ],  # overflorian, phyks, francois2
        "camera_models": ["LG-R105"],
        "osm_file": "paris.osm",
    },
    "nantes": {
        "bbox": BoundaryBox([-1.585839, 47.198289][::-1], [-1.51318, 47.236161][::-1]),
        "owners": [
            "jGdq3CL-9N-Esvj3mtCWew",
            "s-j5BH9JRIzsgORgaJF3aA",
        ],  # c_mobilite, cartocite
        "osm_file": "nantes.osm",
    },
    "toulouse": {
        "bbox": BoundaryBox([1.429457, 43.591434][::-1], [1.456653, 43.61343][::-1]),
        "owners": ["MNkhq6MCoPsdQNGTMh3qsQ"],  # tyndare
        "osm_file": "toulouse.osm",
    },
    "vilnius": {
        "bbox": BoundaryBox([25.258633, 54.672956][::-1], [25.296094, 54.696755][::-1]),
        "owners": ["bClduFF6Gq16cfwCdhWivw", "u5ukBseATUS8jUbtE43fcO"],  # kedas, vms
        "osm_file": "vilnius.osm",
    },
    "helsinki": {
        "bbox": BoundaryBox(
            [24.8975480117, 60.1449128318][::-1], [24.9816543235, 60.1770977471][::-1]
        ),
        "camera_types": ["spherical", "equirectangular"],
        "osm_file": "helsinki.osm",
    },
    "milan": {
        "bbox": BoundaryBox(
            [9.1732723899, 45.4810977947][::-1],
            [9.2255987917, 45.5284238563][::-1],
        ),
        "camera_types": ["spherical", "equirectangular"],
        "osm_file": "milan.osm",
    },
    "avignon": {
        "bbox": BoundaryBox(
            [4.7887045302, 43.9416178156][::-1], [4.8227015622, 43.9584848909][::-1]
        ),
        "camera_types": ["spherical", "equirectangular"],
        "osm_file": "avignon.osm",
    },
    "paris": {
        "bbox": BoundaryBox([2.306823, 48.833827][::-1], [2.39067, 48.889335][::-1]),
        "camera_types": ["spherical", "equirectangular"],
        "osm_file": "paris.osm",
    },
}


cfg = OmegaConf.create(
    {
        "max_image_size": 512,
        "do_legacy_pano_offset": True,
        "min_dist_between_keyframes": 4,
        "tiling": {
            "tile_size": 128,
            "margin": 128,
            "ppm": 2,
        },
    }
)


def get_pano_offset(image_info: dict, do_legacy: bool = False) -> float:
    if do_legacy:
        seed = int(image_info["sfm_cluster"]["id"])
    else:
        seed = image_info["sequence"].__hash__()
    seed = seed % (2**32 - 1)
    return np.random.RandomState(seed).uniform(-45, 45)


def process_shot(
    shot: Shot, info: dict, image_path: Path, output_dir: Path, cfg: DictConfig
) -> List[Shot]:
    if not image_path.exists():
        return None

    image_orig = cv2.imread(str(image_path))
    max_size = cfg.max_image_size
    pano_offset = None

    camera = shot.camera
    camera.width, camera.height = image_orig.shape[:2][::-1]
    if camera.is_panorama(camera.projection_type):
        camera_new = perspective_camera_from_pano(camera, max_size)
        undistorter = PanoramaUndistorter(camera, camera_new)
        pano_offset = get_pano_offset(info, cfg.do_legacy_pano_offset)
    elif camera.projection_type in ["fisheye", "perspective"]:
        if camera.projection_type == "fisheye":
            camera_new = perspective_camera_from_fisheye(camera)
        else:
            camera_new = perspective_camera_from_perspective(camera)
        camera_new = scale_camera(camera_new, max_size)
        camera_new.id = camera.id + "_undistorted"
        undistorter = CameraUndistorter(camera, camera_new)
    else:
        raise NotImplementedError(camera.projection_type)

    shots_undist, images_undist = undistort_shot(
        image_orig, shot, undistorter, pano_offset
    )
    for shot, image in zip(shots_undist, images_undist):
        cv2.imwrite(str(output_dir / f"{shot.id}.jpg"), image)

    return shots_undist


def pack_shot_dict(shot: Shot, info: dict) -> dict:
    latlong = info["computed_geometry"]["coordinates"][::-1]
    latlong_gps = info["geometry"]["coordinates"][::-1]
    w_p_c = shot.pose.get_origin()
    w_r_c = shot.pose.get_R_cam_to_world()
    rpy = decompose_rotmat(w_r_c)
    return dict(
        camera_id=shot.camera.id,
        latlong=latlong,
        t_c2w=w_p_c,
        R_c2w=w_r_c,
        roll_pitch_yaw=rpy,
        capture_time=info["captured_at"],
        gps_position=np.r_[latlong_gps, info["altitude"]],
        compass_angle=info["compass_angle"],
        chunk_id=int(info["sfm_cluster"]["id"]),
    )


def pack_camera_dict(camera: Camera) -> dict:
    assert camera.projection_type == "perspective"
    K = camera.get_K_in_pixel_coordinates(camera.width, camera.height)
    return dict(
        id=camera.id,
        model="PINHOLE",
        width=camera.width,
        height=camera.height,
        params=K[[0, 1, 0, 1], [0, 1, 2, 2]],
    )


def process_sequence(
    image_ids: List[int],
    image_infos: dict,
    projection: Projection,
    cfg: DictConfig,
    raw_image_dir: Path,
    out_image_dir: Path,
):
    shots = []
    image_ids = sorted(image_ids, key=lambda i: image_infos[i]["captured_at"])
    for i in image_ids:
        _, shot = opensfm_shot_from_info(image_infos[i], projection)
        shots.append(shot)
    if not shots:
        return {}

    shot_idxs = keyframe_selection(shots, min_dist=cfg.min_dist_between_keyframes)
    shots = [shots[i] for i in shot_idxs]

    shots_out = thread_map(
        lambda shot: process_shot(
            shot,
            image_infos[int(shot.id)],
            raw_image_dir / image_filename.format(image_id=shot.id),
            out_image_dir,
            cfg,
        ),
        shots,
        disable=True,
    )
    shots_out = [(i, s) for i, ss in enumerate(shots_out) for s in ss if ss is not None]

    dump = {}
    for index, shot in shots_out:
        i, suffix = shot.id.rsplit("_", 1)
        info = image_infos[int(i)]
        seq_id = info["sequence"]
        is_pano = not suffix.endswith("undistorted")
        if is_pano:
            seq_id += f"_{suffix}"
        if seq_id not in dump:
            dump[seq_id] = dict(views={}, cameras={})

        view = pack_shot_dict(shot, info)
        view["index"] = index
        dump[seq_id]["views"][shot.id] = view
        dump[seq_id]["cameras"][shot.camera.id] = pack_camera_dict(shot.camera)
    return dump


def process_location(
    location: str,
    data_dir: Path,
    split_path: Path,
    token: str,
    generate_tiles: bool = False,
):
    params = location_to_params[location]
    bbox = params["bbox"]
    projection = Projection(*bbox.center)

    splits = json.loads(split_path.read_text())
    image_ids = [i for split in splits.values() for i in split[location]]

    loc_dir = data_dir / location
    infos_dir = loc_dir / "image_infos"
    raw_image_dir = loc_dir / "images_raw"
    out_image_dir = loc_dir / "images"
    for d in (infos_dir, raw_image_dir, out_image_dir):
        d.mkdir(parents=True, exist_ok=True)

    downloader = MapillaryDownloader(token)
    loop = asyncio.get_event_loop()

    logger.info("Fetching metadata for all images.")
    image_infos, num_fail = loop.run_until_complete(
        fetch_image_infos(image_ids, downloader, infos_dir)
    )
    logger.info("%d failures (%.1f%%).", num_fail, 100 * num_fail / len(image_ids))

    logger.info("Fetching image pixels.")
    image_urls = [(i, info["thumb_2048_url"]) for i, info in image_infos.items()]
    num_fail = loop.run_until_complete(
        fetch_images_pixels(image_urls, downloader, raw_image_dir)
    )
    logger.info("%d failures (%.1f%%).", num_fail, 100 * num_fail / len(image_urls))

    seq_to_image_ids = defaultdict(list)
    for i, info in image_infos.items():
        seq_to_image_ids[info["sequence"]].append(i)
    seq_to_image_ids = dict(seq_to_image_ids)

    dump = {}
    for seq_image_ids in tqdm(seq_to_image_ids.values()):
        dump.update(
            process_sequence(
                seq_image_ids,
                image_infos,
                projection,
                cfg,
                raw_image_dir,
                out_image_dir,
            )
        )
    write_json(loc_dir / "dump.json", dump)

    # Get the view locations
    view_ids = []
    views_latlon = []
    for seq in dump:
        for view_id, view in dump[seq]["views"].items():
            view_ids.append(view_id)
            views_latlon.append(view["latlong"])
    views_latlon = np.stack(views_latlon)
    view_ids = np.array(view_ids)
    views_xy = projection.project(views_latlon)

    tiles_path = loc_dir / MapillaryDataModule.default_cfg["tiles_filename"]
    if generate_tiles:
        logger.info("Creating the map tiles.")
        bbox_data = BoundaryBox(views_xy.min(0), views_xy.max(0))
        bbox_tiling = bbox_data + cfg.tiling.margin
        osm_dir = data_dir / "osm"
        osm_path = osm_dir / params["osm_file"]
        if not osm_path.exists():
            logger.info("Downloading OSM raw data.")
            download_file(DATA_URL + f"/osm/{params['osm_file']}", osm_path)
        if not osm_path.exists():
            raise FileNotFoundError(f"Cannot find OSM data file {osm_path}.")
        tile_manager = TileManager.from_bbox(
            projection,
            bbox_tiling,
            cfg.tiling.ppm,
            tile_size=cfg.tiling.tile_size,
            path=osm_path,
        )
        tile_manager.save(tiles_path)
    else:
        logger.info("Downloading pre-generated map tiles.")
        download_file(DATA_URL + f"/tiles/{location}.pkl", tiles_path)

    # Visualize the data split
    plotter = GeoPlotter()
    view_ids_val = set(splits["val"][location])
    is_val = np.array([int(i.rsplit("_", 1)[0]) in view_ids_val for i in view_ids])
    plotter.points(views_latlon[~is_val], "red", view_ids[~is_val], "train")
    plotter.points(views_latlon[is_val], "green", view_ids[is_val], "val")
    plotter.bbox(bbox, "blue", "query bounding box")
    plotter.bbox(projection.unproject(bbox_tiling), "black", "tiling bounding box")
    geo_viz_path = loc_dir / f"split_{location}.html"
    plotter.fig.write_html(geo_viz_path)
    logger.info("Wrote split visualization to %s.", geo_viz_path)

    shutil.rmtree(raw_image_dir)
    logger.info("Done processing for location %s.", location)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--locations", type=str, nargs="+", default=list(location_to_params)
    )
    parser.add_argument("--split_filename", type=str, default="splits_MGL_13loc.json")
    parser.add_argument("--token", type=str, required=True)
    parser.add_argument(
        "--data_dir", type=Path, default=MapillaryDataModule.default_cfg["data_dir"]
    )
    parser.add_argument("--generate_tiles", action="store_true")
    args = parser.parse_args()

    args.data_dir.mkdir(exist_ok=True, parents=True)
    shutil.copy(Path(__file__).parent / args.split_filename, args.data_dir)

    for location in args.locations:
        logger.info("Starting processing for location %s.", location)
        process_location(
            location,
            args.data_dir,
            args.data_dir / args.split_filename,
            args.token,
            args.generate_tiles,
        )
