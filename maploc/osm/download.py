# Copyright (c) Meta Platforms, Inc. and affiliates.

import json, math, urllib3
from http.client import responses
from pathlib import Path
from typing import Any, Dict, Optional

import geopandas as gpd

import pandas as pd

from mapbox_vector_tile import decode
from shapely.geometry import shape
from shapely.wkt import loads as parse_wkt

from .. import logger
from ..utils.geo import BoundaryBox

OSM_URL = "https://api.openstreetmap.org/api/0.6/map.json"

FB_TILE_SERVER_URL = (
    # "https://www.internalfb.com/intern/maps/vtp/s1/20250217080099/{z}/{x}/{y}/"
    "https://external.xx.fbcdn.net/maps/vtp/s1/77/{z}/{x}/{y}/?locale=en_US"
)


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
        error = result.info()["error"]
        raise ValueError(f"{result.status} {responses[result.status]}: {error}")

    if cache_path is not None:
        cache_path.write_bytes(result.data)
    return result.json()


def lat_lon_to_tile(lat, lon, zoom):
    """Convert latitude and longitude to tile coordinates."""
    n = 2.0**zoom
    x_tile = int((lon + 180.0) / 360.0 * n)
    y_tile = int(
        (
            1.0
            - (
                math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat)))
                / math.pi
            )
        )
        / 2.0
        * n
    )
    return x_tile, y_tile


def tile_to_quadkey(x_tile, y_tile, zoom):
    """Convert tile coordinates to a quadkey."""
    quadkey = ""
    for i in range(zoom, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if (x_tile & mask) != 0:
            digit += 1
        if (y_tile & mask) != 0:
            digit += 2
        quadkey += str(digit)
    return quadkey


def bounding_box_to_tiles(north, south, east, west, zoom=16):

    print(north, south, east, west, zoom)
    """Generate a list of quadkeys for a bounding box at a given zoom level."""

    # Convert bounding box corners to tile coordinates
    x_tile_min, y_tile_max = lat_lon_to_tile(south, west, zoom)
    x_tile_max, y_tile_min = lat_lon_to_tile(north, east, zoom)

    print(x_tile_min, y_tile_max)
    print(x_tile_max, y_tile_min)

    # Generate quadkeys for all tiles in the bounding box
    tiles = []
    for x_tile in range(x_tile_min, x_tile_max + 1):
        for y_tile in range(y_tile_min, y_tile_max + 1):
            quadkey = tile_to_quadkey(x_tile, y_tile, zoom)
            print(quadkey)
            tiles.append((x_tile, y_tile))

    return tiles


# https://gis.stackexchange.com/questions/401541/decoding-mapbox-vector-tiles/460173#460173
def pixel2deg(xtile, ytile, zoom, xpixel, ypixel, extent=4096):
    xtile = xtile + (xpixel / extent)
    ytile = ytile + ((extent - ypixel) / extent)
    lon_deg = (xtile / 2**zoom) * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / 2**zoom)))
    lat_deg = math.degrees(lat_rad)
    return (lon_deg, lat_deg)


def vector_tiles_to_geodataframe(
    bbox: BoundaryBox,
    zoom: int = 16,
):
    (south, west), (north, east) = bbox.min_, bbox.max_

    print("here")

    tiles = bounding_box_to_tiles(
        north=north, south=south, east=east, west=west, zoom=zoom
    )

    features = []
    _columns = ["group", "label", "geometry"]

    for t in tiles:
        result = urllib3.request(
            "GET",
            FB_TILE_SERVER_URL.format(x=t[0], y=t[1], z=zoom),
            fields={},
            timeout=10,
        )

        if result.status != 200:
            error = result.info()["error"]
            raise ValueError(f"{result.status} {responses[result.status]}: {error}")

        decoded_tile = decode(
            tile=result.data,
            default_options={
                "transformer": lambda x, y: pixel2deg(t[0], t[1], zoom, x, y)
            },
        )

        # Clip Dataframe and add relevant columns based on properties
        if decoded_tile.get("building"):
            buildings = gpd.GeoDataFrame(
                [
                    f.get("properties")
                    for f in decoded_tile.get("building").get("features")
                ],
                geometry=[
                    shape(f.get("geometry"))
                    for f in decoded_tile.get("building").get("features")
                ],
            )

            if "isDetail" in buildings.columns:
                buildings = buildings[buildings.isDetail != "true"]

            buildings = buildings.clip(mask=[west, south, east, north])
            buildings["group"] = "building"
            buildings["label"] = "none"

            features.append(buildings[pd.notnull(buildings.group)][_columns])

        if decoded_tile.get("road"):
            roads = gpd.GeoDataFrame(
                [f.get("properties") for f in decoded_tile.get("road").get("features")],
                geometry=[
                    shape(f.get("geometry"))
                    for f in decoded_tile.get("road").get("features")
                ],
            )

            roads = roads.clip(mask=[west, south, east, north])
            roads["group"] = roads["class"].apply(
                lambda _cls: "path" if _cls in {"pedestrian"} else "road"
            )
            roads["label"] = roads["class"]

            features.append(roads[pd.notnull(roads.group)][_columns])

        if decoded_tile.get("landuse"):
            landuse = gpd.GeoDataFrame(
                [
                    f.get("properties")
                    for f in decoded_tile.get("landuse").get("features")
                ],
                geometry=[
                    shape(f.get("geometry"))
                    for f in decoded_tile.get("landuse").get("features")
                ],
            )

            landuse = landuse.clip(mask=[west, south, east, north])
            landuse["group"] = landuse["class"].apply(
                lambda _cls: "grass" if _cls in {"greenspace"} else None
            )
            landuse["label"] = landuse["class"]

            features.append(landuse[pd.notnull(landuse.group)][_columns])

    return pd.concat(features)


def earth_to_geodataframe(
    bbox: BoundaryBox,
):
    (south, west), (north, east) = bbox.min_, bbox.max_

    _columns = ["group", "label", "geometry"]

    # Query the earth table ()
    # Stub this with notebook.json for now
    df = pd.read_json("assets/notebook.json")
    df["geometry"] = df.wkt.apply(parse_wkt)
    gdf = gpd.GeoDataFrame(df, geometry="geometry")

    gdf["group"] = None
    # gdf.apply(
    # lambda row: parse_osm_tags_to_group(json.loads(row.tags), row.geometry), axis=1
    # )

    gdf["label"] = None

    filtered = gdf[pd.notnull(gdf.group)][_columns]

    clipped = filtered.clip(mask=[west, south, east, north])

    print(clipped)

    return clipped
