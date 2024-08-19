# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import subprocess
from http.client import responses
from pathlib import Path
from typing import Any, Dict, Optional

import shapely
import urllib3

from .. import logger
from ..utils.geo import BoundaryBox

OSM_URL = "https://api.openstreetmap.org/api/0.6/map.json"


def get_osm(
    boundary_box: BoundaryBox,
    cache_path: Optional[Path] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Fetch OSM data using the OSM API. Suitable only for small areas."""
    if not overwrite and cache_path is not None and cache_path.is_file():
        return json.loads(cache_path.read_text())

    (bottom, left), (top, right) = boundary_box.min_, boundary_box.max_
    query = {"bbox": f"{left},{bottom},{right},{top}"}

    logger.info("Calling the OpenStreetMap API...")
    result = urllib3.request("GET", OSM_URL, fields=query, timeout=10)
    if result.status != 200:
        error = result.info()["error"]
        raise ValueError(f"{result.status} {responses[result.status]}: {error}")

    if cache_path is not None:
        cache_path.write_bytes(result.data)
    return result.json()


def get_geofabrik_index() -> Dict[str, Any]:
    """Fetch the index of all regions served by Geofabrik."""
    result = urllib3.request(
        "GET", "https://download.geofabrik.de/index-v1.json", timeout=10
    )
    if result.status != 200:
        error = result.info()["error"]
        raise ValueError(f"{result.status} {responses[result.status]}: {error}")
    return json.loads(result.data)


def get_geofabrik_url(bbox: BoundaryBox) -> str:
    """Find the smallest Geofabrik region file that includes a given area."""
    gf = get_geofabrik_index()
    best_region = None
    best_area = float("inf")
    query_poly = shapely.box(*bbox.min_[::-1], *bbox.max_[::-1])
    for i, region in enumerate(gf["features"]):
        coords = region["geometry"]["coordinates"]
        # fix the polygon format
        coords = [c if len(c[0]) < 2 else (c[0], c[1:]) for c in coords]
        poly = shapely.MultiPolygon([shapely.Polygon(*c) for c in coords])
        if poly.contains(query_poly):
            area = poly.area
            if area < best_area:
                best_area = area
                best_region = region
    return best_region["properties"]["urls"]["pbf"]


def convert_osm_file(bbox: BoundaryBox, input_path: Path, output_path: Path):
    """Convert and crop a binary .pbf file to an .osm XML file."""
    bbox_str = ",".join(map(str, (*bbox.min_[::-1], *bbox.max_[::-1])))
    cmd = [
        "osmium",
        "extract",
        "-s",
        "smart",
        "-b",
        bbox_str,
        input_path.as_posix(),
        "-o",
        output_path.as_posix(),
        "--overwrite",
    ]
    subprocess.run(cmd, check=True)
