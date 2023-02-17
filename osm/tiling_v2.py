# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import io
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import fsspec
import numpy as np
from PIL import Image

from ..utils.geo import BoundaryBox, Projection
from .data import MapData
from .download import get_osm
from .index import MapIndex
from .parser import Groups
from .raster import Canvas, render_raster_map, render_raster_masks
from .reader import OSMData


def bbox_to_slice(bbox: BoundaryBox, canvas: Canvas):
    uv_min = np.ceil(canvas.to_uv(bbox.min_)).astype(int)
    uv_max = np.ceil(canvas.to_uv(bbox.max_)).astype(int)
    slice_ = (slice(uv_max[1], uv_min[1]), slice(uv_min[0], uv_max[0]))
    return slice_


def round_bbox(bbox: BoundaryBox, origin: np.ndarray, ppm: int):
    bbox = bbox.translate(-origin)
    bbox = BoundaryBox(np.round(bbox.min_ * ppm) / ppm, np.round(bbox.max_ * ppm) / ppm)
    return bbox.translate(origin)


class TileManager:
    def __init__(
        self,
        tiles: Dict,
        bbox: BoundaryBox,
        tile_size: int,
        ppm: int,
        projection: Projection,
        groups: Dict[str, List[str]],
        map_data: Optional[MapData] = None,
    ):
        self.origin = bbox.min_
        self.bbox = bbox
        self.tiles = tiles
        self.tile_size = tile_size
        self.ppm = ppm
        self.projection = projection
        self.groups = groups
        self.map_data = map_data
        assert np.all(tiles[0, 0].bbox.min_ == self.origin)
        for tile in tiles.values():
            assert bbox.contains(tile.bbox)

    @classmethod
    def from_bbox(
        cls,
        path: Path,
        projection: Projection,
        bbox: BoundaryBox,
        tile_size: int,
        ppm: int,
    ):
        bbox_osm = projection.unproject(bbox)
        if path.is_file():
            osm = OSMData.from_file(path)
            if osm.box is not None:
                assert osm.box.contains(bbox_osm)
        else:
            path = path / f"{bbox_osm}.json"
            osm = OSMData.from_dict(get_osm(bbox_osm, path))

        osm.add_xy_to_nodes(projection)
        map_data = MapData.from_osm(osm)
        map_index = MapIndex(map_data)

        bounds_x, bounds_y = [
            np.r_[np.arange(min_, max_, tile_size), max_]
            for min_, max_ in zip(bbox.min_, bbox.max_)
        ]
        bbox_tiles = {}
        for i, xmin in enumerate(bounds_x[:-1]):
            for j, ymin in enumerate(bounds_y[:-1]):
                bbox_tiles[i, j] = BoundaryBox(
                    [xmin, ymin], [bounds_x[i + 1], bounds_y[j + 1]]
                )

        tiles = {}
        for ij, bbox_tile in bbox_tiles.items():
            canvas = Canvas(bbox_tile, ppm)
            nodes, lines, areas = map_index.query(bbox_tile)
            masks = render_raster_masks(nodes, lines, areas, canvas)
            canvas.raster = render_raster_map(masks)
            tiles[ij] = canvas

        groups = {k: v for k, v in vars(Groups).items() if not k.startswith("__")}

        return cls(tiles, bbox, tile_size, ppm, projection, groups, map_data)

    def query(self, bbox: BoundaryBox) -> Canvas:
        bbox = round_bbox(bbox, self.bbox.min_, self.ppm)
        canvas = Canvas(bbox, self.ppm)
        raster = np.zeros((3, canvas.h, canvas.w), np.uint8)

        bbox_all = bbox & self.bbox
        ij_min = np.floor((bbox_all.min_ - self.origin) / self.tile_size).astype(int)
        ij_max = np.ceil((bbox_all.max_ - self.origin) / self.tile_size).astype(int) - 1
        for i in range(ij_min[0], ij_max[0] + 1):
            for j in range(ij_min[1], ij_max[1] + 1):
                tile = self.tiles[i, j]
                bbox_select = tile.bbox & bbox
                slice_query = bbox_to_slice(bbox_select, canvas)
                slice_tile = bbox_to_slice(bbox_select, tile)
                raster[(slice(None),) + slice_query] = tile.raster[
                    (slice(None),) + slice_tile
                ]
        canvas.raster = raster
        return canvas

    def save(self, path: Path):
        dump = {
            "bbox": self.bbox.format(),
            "tile_size": self.tile_size,
            "ppm": self.ppm,
            "groups": self.groups,
            "tiles_bbox": {},
            "tiles_raster": {},
        }
        if self.projection is not None:
            dump["epsg"] = self.projection.epsg
        for ij, canvas in self.tiles.items():
            dump["tiles_bbox"][ij] = canvas.bbox.format()
            raster_bytes = io.BytesIO()
            raster = Image.fromarray(canvas.raster.transpose(1, 2, 0).astype(np.uint8))
            raster.save(raster_bytes, format="PNG")
            dump["tiles_raster"][ij] = raster_bytes
        with open(path, "wb") as fp:
            pickle.dump(dump, fp)

    @classmethod
    def load(cls, path: Path):
        with fsspec.open(str(path), "rb") as fp:
            dump = pickle.load(fp)
        tiles = {}
        for ij, bbox in dump["tiles_bbox"].items():
            tiles[ij] = Canvas(BoundaryBox.from_string(bbox), dump["ppm"])
            raster = np.asarray(Image.open(dump["tiles_raster"][ij]))
            tiles[ij].raster = raster.transpose(2, 0, 1).copy()
        return cls(
            tiles,
            BoundaryBox.from_string(dump["bbox"]),
            dump["tile_size"],
            dump["ppm"],
            Projection(dump["epsg"]) if "epsg" in dump else None,
            dump["groups"],
        )
