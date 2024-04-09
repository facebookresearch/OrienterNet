# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import torchmetrics
from torchmetrics.utilities.data import dim_zero_cat

from maploc.utils.wrappers import Transform2D

from .utils import deg2rad, rotmat2d


def location_error(xy, xy_gt):
    return torch.norm(xy - xy_gt.to(xy), dim=-1)


def angle_error(t, t_gt):
    error = torch.abs(t % 360 - t_gt.to(t) % 360)
    error = torch.minimum(error, 360 - error)
    return error


class Location2DRecall(torchmetrics.MeanMetric):
    def __init__(self, threshold, key="tile_T_cam_max", *args, **kwargs):
        self.threshold = threshold
        self.key = key
        super().__init__(*args, **kwargs)

    def update(self, pred, data):
        if isinstance(pred[self.key], Transform2D):
            xy_p = pred[self.key].t
        else:
            xy_p = pred[self.key]
        xy_gt = data["tile_T_cam"].t
        assert xy_gt.shape == xy_p.shape
        error = location_error(xy_p, xy_gt)
        super().update((error <= self.threshold).float())


class AngleRecall(torchmetrics.MeanMetric):
    def __init__(self, threshold, key="tile_T_cam_max", *args, **kwargs):
        self.threshold = threshold
        self.key = key
        super().__init__(*args, **kwargs)

    def update(self, pred, data):
        error = angle_error(pred[self.key].angle, data["tile_T_cam"].angle)
        super().update((error <= self.threshold).float())


class MeanMetricWithRecall(torchmetrics.Metric):
    full_state_update = True

    def __init__(self):
        super().__init__()
        self.add_state("value", default=[], dist_reduce_fx="cat")

    def compute(self):
        return dim_zero_cat(self.value).mean(0)

    def get_errors(self):
        return dim_zero_cat(self.value)

    def recall(self, thresholds):
        error = self.get_errors()
        thresholds = error.new_tensor(thresholds)
        return (error.unsqueeze(-1) < thresholds).float().mean(0) * 100


class AngleError(MeanMetricWithRecall):
    def __init__(self, key):
        super().__init__()
        self.key = key

    def update(self, pred, data):
        value = angle_error(pred[self.key].angle, data["tile_T_cam"].angle)
        if value.numel():
            self.value.append(value)


class Location2DError(MeanMetricWithRecall):
    def __init__(self, key):
        super().__init__()
        self.key = key

    def update(self, pred, data):
        if isinstance(pred[self.key], Transform2D):
            xy_p = pred[self.key].t
        else:
            xy_p = pred[self.key]

        xy_gt = data["tile_T_cam"].t
        assert xy_gt.shape == xy_p.shape
        value = location_error(xy_p, xy_gt)
        if value.numel():
            self.value.append(value)


class LateralLongitudinalError(MeanMetricWithRecall):
    def __init__(self, key="tile_T_cam_max"):
        super().__init__()
        self.key = key

    def update(self, pred, data):
        yaw = deg2rad(90 - data["tile_T_cam"].angle).squeeze(-1)
        shift = pred[self.key].t - data["tile_T_cam"].t
        shift = (rotmat2d(yaw) @ shift.unsqueeze(-1)).squeeze(-1)
        error = torch.abs(shift)
        value = error.view(-1, 2)
        if value.numel():
            self.value.append(value)
