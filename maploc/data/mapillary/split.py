import argparse
import asyncio
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from omegaconf import DictConfig, OmegaConf

from ... import logger
from ...osm.viz import GeoPlotter
from ...utils.geo import BoundaryBox, Projection
from ...utils.io import write_json
from .config import default_cfg, location_to_params
from .dataset import MapillaryDataModule
from .download import MapillaryDownloader, fetch_image_infos, fetch_image_list


def grid_downsample(xy: np.ndarray, bbox: BoundaryBox, resolution: float) -> np.ndarray:
    assert bbox.contains(xy).all()
    extent = bbox.max_ - bbox.min_
    size = np.ceil(extent / resolution).astype(int)
    grid = np.full(size, -1)
    idx = np.floor((xy - bbox.min_) / resolution).astype(int)
    grid[tuple(idx.T)] = np.arange(len(xy))
    indices = grid[grid >= 0]
    return indices


def find_validation_bbox(
    xy: np.ndarray, target_num, size_upper_bound=500
) -> BoundaryBox:
    # Find the centroid of all points
    center = np.median(xy, 0)
    dist = np.linalg.norm(xy - center[None], axis=1)
    center = xy[np.argmin(dist)]

    bbox = BoundaryBox(center - size_upper_bound, center + size_upper_bound)
    mask = bbox.contains(xy)
    dist = np.abs(xy[mask] - center).max(-1)
    dist.sort()
    thresh = dist[target_num]
    bbox_val = BoundaryBox(center - thresh, center + thresh)
    return bbox_val


def process_location(
    output_path: Path,
    token: str,
    cfg: DictConfig,
    query_bbox: BoundaryBox,
    filters: Dict[str, Any],
    bbox_val: Optional[BoundaryBox] = None,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    downloader = MapillaryDownloader(token)
    loop = asyncio.get_event_loop()
    projection = Projection(*query_bbox.center)
    bbox_local = projection.project(query_bbox)

    logger.info("Fetching the list of image with filter: %s", filters)
    image_ids, bboxes = loop.run_until_complete(
        fetch_image_list(query_bbox, downloader, **filters)
    )
    if not image_ids:
        raise ValueError("Could not find any image!")
    logger.info("Found %d images.", len(image_ids))

    logger.info("Fetching the image coordinates.")
    infos, num_fail = loop.run_until_complete(
        fetch_image_infos(image_ids, downloader, fields=["computed_geometry"])
    )
    logger.info("%d failures (%.1f%%).", num_fail, 100 * num_fail / len(image_ids))

    # discard images that don't have coordinates available
    image_ids = np.array([i for i in infos if "computed_geometry" in infos[i]])
    image_ids.sort()

    # discard images outside of the query bbox
    latlon = np.array(
        [infos[i]["computed_geometry"]["coordinates"][::-1] for i in image_ids]
    )
    xy = projection.project(latlon)
    valid = bbox_local.contains(xy)
    image_ids = image_ids[valid]
    latlon = latlon[valid]
    xy = xy[valid]

    # downsample the images with a grid
    indices = grid_downsample(xy, bbox_local, cfg.downsampling_resolution_meters)
    image_ids = image_ids[indices]
    latlon = latlon[indices]
    xy = xy[indices]
    logger.info("Filtered down to %d images.", len(image_ids))

    if bbox_val is None:
        bbox_val_local = find_validation_bbox(xy, cfg.target_num_val_images)
        bbox_val = projection.unproject(bbox_val_local)
    else:
        bbox_val_local = projection.project(bbox_val)
    logger.info("Using validation bounding box: %s.", bbox_val)
    indices_val = np.where(bbox_val_local.contains(xy))[0]
    bbox_not_train = bbox_val_local + cfg.val_train_margin_meters
    indices_train = np.where(~bbox_not_train.contains(xy))[0]
    if len(indices_train) > cfg.max_num_train_images:
        indices_subsample = np.random.RandomState(0).choice(
            len(indices_train), cfg.max_num_train_images
        )
        indices_train = indices_train[indices_subsample]
    logger.info(
        "Resulting split: %d val and %d train images.",
        len(indices_val),
        len(indices_train),
    )

    splits = {
        "val": image_ids[indices_val].tolist(),
        "train": image_ids[indices_train].tolist(),
    }
    write_json(output_path, splits)

    # Visualize the data split
    plotter = GeoPlotter()
    plotter.points(latlon[indices_train], "red", image_ids[indices_train], "train")
    plotter.points(latlon[indices_val], "green", image_ids[indices_val], "val")
    plotter.bbox(query_bbox, "blue", "query bounding box")
    plotter.bbox(bbox_val, "green", "validation bounding box")
    plotter.bbox(projection.unproject(bbox_not_train), "black", "margin bounding box")
    geo_viz_path = f"{output_path}_viz.html"
    plotter.fig.write_html(geo_viz_path)
    logger.info("Wrote split visualization to %s.", geo_viz_path)


def main(args: argparse.Namespace):
    args.data_dir.mkdir(exist_ok=True, parents=True)
    cfg = OmegaConf.merge(default_cfg, OmegaConf.from_cli(args.dotlist))
    for location in args.locations:
        output_path = args.data_dir / args.output_filename.format(scene=location)
        if output_path.exists() and not args.overwrite:
            logger.info("Skipping processing for location %s.", location)
            continue
        logger.info("Starting processing for location %s.", location)
        params = location_to_params[location]
        process_location(
            output_path,
            args.token,
            cfg,
            params["bbox"],
            params["filters"] | {"start_captured_at": args.min_capture_date},
            None if args.force_auto_val_bbox else params.get("bbox_val"),
        )
        logger.info("Done processing for location %s.", location)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--locations", type=str, nargs="+", default=list(location_to_params)
    )
    parser.add_argument("--token", type=str, required=True)
    parser.add_argument("--min_capture_date", type=str, default="2015-01-01T00:00:00Z")
    parser.add_argument(
        "--output_filename", type=str, default="splits_MGL_v2_{scene}.json"
    )
    parser.add_argument(
        "--data_dir", type=Path, default=MapillaryDataModule.default_cfg["data_dir"]
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--force_auto_val_bbox", action="store_true")
    parser.add_argument("dotlist", nargs="*")
    main(parser.parse_args())
