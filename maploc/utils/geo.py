# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import Union

import numpy as np
import torch

from .. import logger
from .geo_opensfm import TopocentricConverter


class BoundaryBox:
    def __init__(self, min_: np.ndarray, max_: np.ndarray):
        self.min_ = np.asarray(min_)
        self.max_ = np.asarray(max_)
        assert np.all(self.min_ <= self.max_)

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
    def __init__(self, lat, lon, alt=0, max_extent=25e3):
        # The approximation error is |L - radius * tan(L / radius)|
        # and is around 13cm for L=25km.
        self.latlonalt = (lat, lon, alt)
        self.converter = TopocentricConverter(lat, lon, alt)
        min_ = self.converter.to_lla(*(-max_extent,) * 2, 0)[:2]
        max_ = self.converter.to_lla(*(max_extent,) * 2, 0)[:2]
        self.bounds = BoundaryBox(min_, max_)

    @classmethod
    def from_points(cls, all_latlon):
        assert all_latlon.shape[-1] == 2
        all_latlon = all_latlon.reshape(-1, 2)
        latlon_mid = (all_latlon.min(0) + all_latlon.max(0)) / 2
        return cls(*latlon_mid)

    def check_bbox(self, bbox: BoundaryBox):
        if self.bounds is not None and not self.bounds.contains(bbox):
            raise ValueError(
                f"Bbox {bbox.format()} is not contained in "
                f"projection with bounds {self.bounds.format()}."
            )

    def project(self, geo, return_z=False):
        if isinstance(geo, BoundaryBox):
            return BoundaryBox(*self.project(np.stack([geo.min_, geo.max_])))
        geo = np.asarray(geo)
        assert geo.shape[-1] in (2, 3)
        if self.bounds is not None:
            if not np.all(self.bounds.contains(geo[..., :2])):
                raise ValueError(
                    f"Points {geo} are out of the valid bounds "
                    f"{self.bounds.format()}."
                )
        lat, lon = geo[..., 0], geo[..., 1]
        if geo.shape[-1] == 3:
            alt = geo[..., -1]
        else:
            alt = np.zeros_like(lat)
        x, y, z = self.converter.to_topocentric(lat, lon, alt)
        return np.stack([x, y] + ([z] if return_z else []), -1)

    def unproject(self, xy, return_z=False):
        if isinstance(xy, BoundaryBox):
            return BoundaryBox(*self.unproject(np.stack([xy.min_, xy.max_])))
        xy = np.asarray(xy)
        x, y = xy[..., 0], xy[..., 1]
        if xy.shape[-1] == 3:
            z = xy[..., -1]
        else:
            z = np.zeros_like(x)
        lat, lon, alt = self.converter.to_lla(x, y, z)
        return np.stack([lat, lon] + ([alt] if return_z else []), -1)
