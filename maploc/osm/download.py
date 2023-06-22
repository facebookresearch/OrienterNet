# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
from pathlib import Path
from typing import Dict, Optional

import urllib3

from .. import logger
from ..utils.geo import BoundaryBox


def get_osm(
    boundary_box: BoundaryBox,
    cache_path: Optional[Path] = None,
    overwrite: bool = False,
) -> str:
    if not overwrite and cache_path is not None and cache_path.is_file():
        with cache_path.open() as fp:
            return json.load(fp)

    (bottom, left), (top, right) = boundary_box.min_, boundary_box.max_
    content: bytes = get_web_data(
        "https://api.openstreetmap.org/api/0.6/map.json",
        {"bbox": f"{left},{bottom},{right},{top}"},
    )

    content_str = content.decode("utf-8")
    if content_str.startswith("You requested too many nodes"):
        raise ValueError(content_str)

    if cache_path is not None:
        with cache_path.open("bw+") as fp:
            fp.write(content)
    return json.loads(content_str)


def get_web_data(address: str, parameters: Dict[str, str]) -> bytes:
    logger.info("Getting %s...", address)
    http = urllib3.PoolManager()
    result = http.request("GET", address, parameters, timeout=10)
    return result.data
