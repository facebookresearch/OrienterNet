# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
from pathlib import Path
from typing import Any, Dict, Optional
from http.client import responses

import urllib3

from .. import logger
from ..utils.geo import BoundaryBox

OSM_URL = "https://api.openstreetmap.org/api/0.6/map.json"


def get_osm(
    boundary_box: BoundaryBox,
    cache_path: Optional[Path] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    if not overwrite and cache_path is not None and cache_path.is_file():
        return json.loads(cache_path.read_text())

    (bottom, left), (top, right) = boundary_box.min_, boundary_box.max_
    query = {"bbox": f"{left},{bottom},{right},{top}"}

    logger.info("Calling the OpenStreetMap API...")
    result = urllib3.request("GET", OSM_URL, fields=query, timeout=10)
    if result.status != 200:
        error = result.info()['error']
        raise ValueError(f"{result.status} {responses[result.status]}: {error}")

    if cache_path is not None:
        cache_path.write_bytes(result.data)
    return result.json()
