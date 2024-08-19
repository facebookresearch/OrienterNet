# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import asyncio
import json
import shutil
from collections import defaultdict
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
from omegaconf import DictConfig, OmegaConf
from opensfm.pygeometry import Camera
from opensfm.pymap import Shot
from opensfm.undistort import (
    perspective_camera_from_fisheye,
    perspective_camera_from_perspective,
)
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

from ... import logger
from ...osm.download import convert_osm_file, get_geofabrik_url
from ...osm.tiling import TileManager
from ...osm.viz import GeoPlotter
from ...utils.geo import BoundaryBox, Projection
from ...utils.io import DATA_URL, download_file, write_json
from ..utils import decompose_rotmat
from .config import default_cfg, location_to_params
from .dataset import MapillaryDataModule
from .download import (
    MapillaryDownloader,
    fetch_image_infos,
    fetch_images_pixels,
    image_filename,
    opensfm_shot_from_info,
)
from .utils import (
    CameraUndistorter,
    PanoramaUndistorter,
    keyframe_selection,
    perspective_camera_from_pano,
    scale_camera,
    undistort_shot,
)

DATA_FILENAME = "dump.json"


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
    shots_out = [(i, s) for i, ss in enumerate(shots_out) if ss is not None for s in ss]

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
    output_dir: Path,
    bbox: BoundaryBox,
    splits: Dict[str, Sequence[int]],
    token: str,
    cfg: DictConfig,
):
    projection = Projection(*bbox.center)
    image_ids = [i for split in splits.values() for i in split]

    infos_dir = output_dir / "image_infos"
    raw_image_dir = output_dir / "images_raw"
    out_image_dir = output_dir / "images"
    for d in (infos_dir, raw_image_dir, out_image_dir):
        d.mkdir(parents=True, exist_ok=True)

    downloader = MapillaryDownloader(token)
    loop = asyncio.get_event_loop()

    logger.info("Fetching metadata for all images.")
    image_infos, num_fail = loop.run_until_complete(
        fetch_image_infos(image_ids, downloader, dir_=infos_dir)
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
    write_json(output_dir / DATA_FILENAME, dump)

    shutil.rmtree(raw_image_dir)


class OSMDataSource(Enum):
    PRECOMPUTED = auto()
    CACHED = auto()
    LATEST = auto()


def prepare_osm(
    location: str,
    output_dir: Path,
    bbox: BoundaryBox,
    cfg: DictConfig,
    osm_dir: Path,
    osm_source: OSMDataSource,
    osm_filename: Optional[str] = None,
):
    projection = Projection(*bbox.center)
    dump = json.loads((output_dir / DATA_FILENAME).read_text())
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

    tiles_path = output_dir / MapillaryDataModule.default_cfg["tiles_filename"]
    if osm_source == OSMDataSource.PRECOMPUtED:
        logger.info("Downloading pre-computed map tiles.")
        download_file(DATA_URL + f"/tiles/{location}.pkl", tiles_path)
        tile_manager = TileManager.load(tiles_path)
    else:
        logger.info("Creating the map tiles.")
        bbox_data = BoundaryBox(views_xy.min(0), views_xy.max(0))
        bbox_tiling = bbox_data + cfg.tiling.margin
        osm_filename = osm_filename or f"{location}.osm"
        osm_path = osm_dir / osm_filename
        if osm_source == OSMDataSource.CACHED:
            if not osm_path.exists():
                logger.info("Downloading OSM raw data.")
                download_file(DATA_URL + f"/osm/{osm_filename}", osm_path)
            if not osm_path.exists():
                raise FileNotFoundError(f"Cannot find OSM data file {osm_path}.")
        elif osm_source == OSMDataSource.LATEST:
            bbox_osm = projection.unproject(bbox_data + 2_000)  # 2 km
            url = get_geofabrik_url(bbox_osm)
            tmp_path = osm_dir / Path(url).name
            download_file(url, tmp_path)
            convert_osm_file(bbox_osm, tmp_path, osm_path)
        else:
            raise NotImplementedError("Unknown source {osm_source}.")
        tile_manager = TileManager.from_bbox(
            projection,
            bbox_tiling,
            cfg.tiling.ppm,
            tile_size=cfg.tiling.tile_size,
            path=osm_path,
        )
        tile_manager.save(tiles_path)

    # Visualize the data split
    plotter = GeoPlotter()
    plotter.points(views_latlon, "red", view_ids, "images")
    plotter.bbox(bbox, "blue", "query bounding box")
    plotter.bbox(
        projection.unproject(tile_manager.bbox), "black", "tiling bounding box"
    )
    geo_viz_path = output_dir / f"viz_data_{location}.html"
    plotter.fig.write_html(geo_viz_path)
    logger.info("Wrote the visualization to %s.", geo_viz_path)


def main(args: argparse.Namespace):
    args.data_dir.mkdir(exist_ok=True, parents=True)

    split_path = args.data_dir / args.split_filename
    maybe_git_split = Path(__file__).parent / args.split_filename
    if maybe_git_split.exists():
        logger.info("Using official split file at %s.", maybe_git_split)
        shutil.copy(maybe_git_split, args.data_dir)

    cfg = OmegaConf.merge(default_cfg, OmegaConf.from_cli(args.dotlist))
    for location in args.locations:
        logger.info("Starting processing for location %s.", location)
        if split_path.exists():
            splits = json.loads(split_path.read_text())
            splits = {split_name: val[location] for split_name, val in splits.items()}
        else:
            split_path_ = Path(str(split_path).format(scene=location))
            if not split_path_.exists():
                raise ValueError(f"Cannot find any split file at path {split_path}.")
            logger.info("Using per-location split file at %s.", split_path_)
            splits = json.loads(split_path_.read_text())

        process_location(
            args.data_dir / location,
            location_to_params[location]["bbox"],
            splits,
            args.token,
            cfg,
        )

        logger.info("Preparing OSM data.")
        prepare_osm(
            location,
            args.data_dir / location,
            location_to_params[location]["bbox"],
            cfg,
            args.data_dir / "osm",
            OSMDataSource[args.osm_source],
            location_to_params[location].get("osm_file"),
        )
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
    parser.add_argument(
        "--osm_source",
        default=OSMDataSource.PRECOMPUTED.name,
        choices=[e.name for e in OSMDataSource],
    )
    parser.add_argument("dotlist", nargs="*")
    main(parser.parse_args())
