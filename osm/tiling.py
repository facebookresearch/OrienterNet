# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import io
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

from .. import logger
from ..utils.geo import BoundaryBox
from .download import get_osm
from .index import MapIndex
from .parser import group_elements, Groups
from .raster import Canvas, render_raster_map, render_raster_masks
from .reader import OSMData
from .viz import plot_locations


def prepare_tiles(
    views: Dict[str, Any],
    proj,
    out_dir: Path,
    tile_radius: float,
    ppm: int,
    viz: bool = False,
    skip_write: bool = False,
    osm_path: Optional[Path] = None,
) -> Tuple[OSMData, Dict[str, Any]]:

    locations = np.stack([v["t_c2w"][:2] for v in views.values()])
    bbox_all = BoundaryBox(locations.min(0), locations.max(0))
    bbox_margin = bbox_all + tile_radius * 2
    if viz:
        plot_locations(locations, bbox_margin)

    bbox_osm = proj.unproject(bbox_margin)
    logger.info("Geo bbox: %s", bbox_osm)

    if osm_path is None:
        osm_path = out_dir / f"{bbox_osm}.json"
        get_osm(bbox_osm, osm_path)
    osm = OSMData.from_file(osm_path)
    if osm.box is not None:
        assert osm.box.contains(bbox_osm)
    osm.add_xy_to_nodes(proj)
    logger.info("OSM data with %d nodes and %d ways", len(osm.nodes), len(osm.ways))

    elem2group = group_elements(osm)
    logger.info("Grouping: %s", [(k, len(v)) for k, v in elem2group.items()])

    map_index = MapIndex(
        osm,
        node_ids=elem2group["node"].keys(),
        way_ids=elem2group["way"].keys() | elem2group["area"].keys(),
    )

    dump = {
        "osm": {
            "bbox_geo": bbox_osm.format(),
        },
        "raster": {
            "tile_radius": tile_radius,
            "ppm": ppm,
            "groups": {
                "areas": Groups.areas,
                "lines": Groups.ways,
                "nodes": Groups.nodes,
            },
        },
    }
    if not skip_write:
        rasters = {}
        for name, view in tqdm(views.items()):
            center = np.array(view["t_c2w"][:2])
            bbox = BoundaryBox(center - tile_radius, center + tile_radius)
            canvas = Canvas(bbox, ppm)
            nodes, ways = map_index.query(bbox)
            masks = render_raster_masks(nodes, ways, canvas, elem2group)
            raster = render_raster_map(masks)

            raster_pil = Image.fromarray(raster.transpose(1, 2, 0).astype(np.uint8))
            raster_bytes = io.BytesIO()
            raster_pil.save(raster_bytes, format="PNG")
            rasters[name] = raster_bytes
        with open(out_dir / "rasters.pkl", "wb") as fp:
            pickle.dump(rasters, fp)

    return osm, dump
