# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from typing import Union

import numpy as np
import pyproj
import torch


class BoundaryBox:
    def __init__(self, min_: np.ndarray, max_: np.ndarray):
        self.min_ = np.asarray(min_)
        self.max_ = np.asarray(max_)
        assert np.all(self.min_ < self.max_)

    @classmethod
    def from_string(cls, string: str):
        return cls(*np.split(np.array(string.split(","), float), 2))

    @property
    def left_top(self):
        return np.stack([self.min_[..., 0], self.max_[..., 1]], -1)

    @property
    def right_bottom(self) -> (np.ndarray, np.ndarray):
        return np.stack([self.max_[..., 0], self.min_[..., 1]], -1)

    @property
    def center(self) -> np.ndarray:
        return (self.min_ + self.max_) / 2

    @property
    def size(self) -> np.ndarray:
        return self.max_ - self.min_

    def translate(self, t: float):
        return self.__class__(self.min_ + t, self.max_ + t)

    def contains(self, xy: Union[np.ndarray, "BoundaryBox"]):
        if isinstance(xy, self.__class__):
            return self.contains(xy.min_) and self.contains(xy.max_)
        return np.all((xy >= self.min_) & (xy <= self.max_), -1)

    def normalize(self, xy):
        min_, max_ = self.min_, self.max_
        if isinstance(xy, torch.Tensor):
            min_ = torch.from_numpy(min_).to(xy)
            max_ = torch.from_numpy(max_).to(xy)
        return (xy - min_) / (max_ - min_)

    def unnormalize(self, xy):
        min_, max_ = self.min_, self.max_
        if isinstance(xy, torch.Tensor):
            min_ = torch.from_numpy(min_).to(xy)
            max_ = torch.from_numpy(max_).to(xy)
        return xy * (max_ - min_) + min_

    def format(self) -> str:
        return ",".join(np.r_[self.min_, self.max_].astype(str))

    def __add__(self, x):
        if isinstance(x, (int, float)):
            return self.__class__(self.min_ - x, self.max_ + x)
        else:
            raise TypeError(f"Cannot add {self.__class__.__name__} to {type(x)}.")

    def __and__(self, other):
        return self.__class__(
            np.maximum(self.min_, other.min_), np.minimum(self.max_, other.max_)
        )

    def __repr__(self):
        return self.format()


class Projection:
    def __init__(self, epsg: Union[str, int]):
        self.crs = pyproj.CRS(epsg)
        self.geo2xy = pyproj.Proj(self.crs)
        self.epsg = epsg

        area = self.crs.area_of_use
        if area is None:
            self.bounds = None
        else:
            bounds = area.bounds
            self.bounds = BoundaryBox([bounds[1], bounds[0]], [bounds[3], bounds[2]])

    @classmethod
    def mercator(cls, all_latlon):
        assert all_latlon.shape[-1] == 2
        all_latlon = all_latlon.reshape(-1, 2)
        latlon_mid = (all_latlon.min(0) + all_latlon.max(0)) / 2
        projection = cls(f"+proj=merc +lat_ts={latlon_mid[0]} +units=m")
        xy_mid = projection.project(latlon_mid)
        projection = cls(
            f"+proj=merc +lat_ts={latlon_mid[0]} +units=m +x_0={-xy_mid[0]} +y_0={-xy_mid[1]}"
        )
        return projection

    def check_bbox(self, bbox: BoundaryBox):
        if self.bounds is not None and not self.bounds.contains(bbox):
            raise ValueError(
                f"Bbox {bbox.format()} is not contained in "
                f"EPSG {self.epsg} with bounds {self.bounds.format()}."
            )

    def project(self, geo):
        if isinstance(geo, BoundaryBox):
            return BoundaryBox(*self.project(np.stack([geo.min_, geo.max_])))
        if self.bounds is not None and not np.all(self.bounds.contains(geo)):
            raise ValueError(
                f"Points {geo} are out of the valid bounds {self.bounds.format()}."
            )

        lat, lon = geo[..., 0], geo[..., 1]
        x, y = self.geo2xy(lon, lat, errcheck=True)
        xy = np.stack([x, y], -1)
        assert np.isfinite(xy).all()
        return xy

    def unproject(self, xy):
        if isinstance(xy, BoundaryBox):
            return BoundaryBox(*self.unproject(np.stack([xy.min_, xy.max_])))

        x, y = xy[..., 0], xy[..., 1]
        lon, lat = self.geo2xy(x, y, inverse=True, errcheck=True)
        geo = np.stack([lat, lon], -1)
        assert np.isfinite(geo).all()
        return geo
