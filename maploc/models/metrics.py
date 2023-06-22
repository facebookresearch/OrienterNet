# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import torchmetrics
from torchmetrics.utilities.data import dim_zero_cat

from .utils import deg2rad, rotmat2d


def location_error(uv, uv_gt, ppm=1):
    return torch.norm(uv - uv_gt.to(uv), dim=-1) / ppm


def angle_error(t, t_gt):
    error = torch.abs(t % 360 - t_gt.to(t) % 360)
    error = torch.minimum(error, 360 - error)
    return error


class Location2DRecall(torchmetrics.MeanMetric):
    def __init__(self, threshold, pixel_per_meter, key="uv_max", *args, **kwargs):
        self.threshold = threshold
        self.ppm = pixel_per_meter
        self.key = key
        super().__init__(*args, **kwargs)

    def update(self, pred, data):
        error = location_error(pred[self.key], data["uv"], self.ppm)
        super().update((error <= self.threshold).float())


class AngleRecall(torchmetrics.MeanMetric):
    def __init__(self, threshold, key="yaw_max", *args, **kwargs):
        self.threshold = threshold
        self.key = key
        super().__init__(*args, **kwargs)

    def update(self, pred, data):
        error = angle_error(pred[self.key], data["roll_pitch_yaw"][..., -1])
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
        value = angle_error(pred[self.key], data["roll_pitch_yaw"][..., -1])
        if value.numel():
            self.value.append(value)


class Location2DError(MeanMetricWithRecall):
    def __init__(self, key, pixel_per_meter):
        super().__init__()
        self.key = key
        self.ppm = pixel_per_meter

    def update(self, pred, data):
        value = location_error(pred[self.key], data["uv"], self.ppm)
        if value.numel():
            self.value.append(value)


class LateralLongitudinalError(MeanMetricWithRecall):
    def __init__(self, pixel_per_meter, key="uv_max"):
        super().__init__()
        self.ppm = pixel_per_meter
        self.key = key

    def update(self, pred, data):
        yaw = deg2rad(data["roll_pitch_yaw"][..., -1])
        shift = (pred[self.key] - data["uv"]) * yaw.new_tensor([-1, 1])
        shift = (rotmat2d(yaw) @ shift.unsqueeze(-1)).squeeze(-1)
        error = torch.abs(shift) / self.ppm
        value = error.view(-1, 2)
        if value.numel():
            self.value.append(value)
