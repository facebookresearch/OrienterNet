# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from ..osm.data import MapData
from ..osm.download import get_osm
from ..osm.raster import Canvas
from ..osm.reader import OSMData
from ..utils.geo import BoundaryBox, Projection
from ..utils.viz_2d import plot_images, plot_keypoints


class BuildingFilter:
    def __init__(
        self,
        bbox: BoundaryBox,
        data: MapData,
        ppm: int = 2,
        margin: int = 1,
    ):
        self.bbox = bbox
        self.canvas = Canvas(self.bbox, ppm=ppm)
        for area in data.areas.values():
            if area.group == "building":
                # TODO: remove buildings that intersect with footways or roads
                self.canvas.draw_multipolygon(area.outers + area.inners)
        pad = int(round(margin * ppm))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (pad * 2 + 1, pad * 2 + 1), (pad, pad)
        )
        self.mask = cv2.erode(self.canvas.raster, kernel) > 0

    @classmethod
    def from_bbox(cls, bbox: BoundaryBox, path: Path, proj: Projection, **kwargs):
        bbox_osm = proj.unproject(bbox)
        osm = OSMData.from_dict(get_osm(bbox_osm, path))
        osm.add_xy_to_nodes(proj)
        map_data = MapData.from_osm(osm)
        return cls(bbox, map_data, **kwargs)

    def in_mask(self, xy: np.ndarray, viz: bool = False) -> np.ndarray:
        masked = np.zeros(len(xy), np.bool)
        valid = self.bbox.contains(xy)
        uv = self.canvas.to_uv(xy)
        uv = np.round(uv).astype(int)
        masked[valid] = self.mask[tuple(uv[valid].T[::-1])]
        if viz:
            plot_images([self.mask])
            plot_keypoints([uv[masked]], colors="r")
            plot_keypoints([uv[~masked]])
            plt.xlim([uv.min(0)[0] - 100, uv.max(0)[0] + 100])
            plt.ylim([uv.max(0)[1] + 100, uv.min(0)[1] - 100])
        return masked
