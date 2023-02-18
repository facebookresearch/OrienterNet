# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import torch

from .base import BaseModel
from .feature_extractor import FeatureExtractor
from .hough_voting import argmax_xy, expectation_xy, log_softmax_spatial
from .map_encoder import MapEncoder
from .metrics import Location2DError, Location2DRecall
from .utils import GlobalPooling


class HeatmapRegressor(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.convs = torch.nn.Sequential(
            torch.nn.Conv2d(2 * dim, dim, (1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(dim, 1, (1, 1)),
        )

    def forward(self, image, map_):
        image = image[..., None, None].tile((1, 1) + map_.shape[-2:])
        inp = torch.cat([map_, image], 1)
        return self.convs(inp).squeeze(1)


class BasicLocalizer(BaseModel):
    default_conf = {
        "image_encoder": "???",
        "map_encoder": "???",
        "latent_dim": "???",
        "pixel_per_meter": 2,
        "heatmap_regressor": False,
    }

    def _init(self, conf):
        self.image_encoder = FeatureExtractor(conf.image_encoder.backbone)
        self.image_pooling = GlobalPooling(kind=conf.image_encoder.pooling)
        self.map_encoder = MapEncoder(conf.map_encoder)
        if conf.heatmap_regressor:
            self.regressor = HeatmapRegressor(conf.latent_dim)

    def _forward(self, data):
        image_features = self.image_encoder(data)["feature_maps"]
        image_descriptor = self.image_pooling(image_features[0])
        map_pred = self.map_encoder(data)
        map_level = 0
        map_features = map_pred["map_features"][map_level]
        if self.conf.heatmap_regressor:
            scores = self.regressor(image_features, map_features)
        else:
            scores = torch.einsum("bd,bdhw->bhw", image_descriptor, map_features)
        log_probs = log_softmax_spatial(scores, dims=2)
        if "log_prior" in map_pred:
            log_probs += map_pred["log_prior"][map_level]
        return {
            "scores": scores,
            "log_probs": log_probs,
            "xy_max": argmax_xy(scores).to(scores),
            "xy_expectation": expectation_xy(log_probs.exp())[0],
            "image_features": image_features,
            "image_descriptor": image_descriptor,
            "map": map_pred,
        }

    def loss(self, pred, data):
        heatmap = pred["log_probs"].unsqueeze(1)  # add channels
        grid = data["xy"][:, None, None]  # add spatial dimensions
        size = grid.new_tensor(heatmap.shape[-2:][::-1])
        grid_norm = (grid + 0.5) / size * 2 - 1
        nll = -torch.nn.functional.grid_sample(heatmap, grid_norm)[..., 0, 0, 0]
        return {"total": nll, "nll": nll}

    def metrics(self):
        return {
            "xy_max_error": Location2DError("xy_max", self.conf.pixel_per_meter),
            "xy_expectation_error": Location2DError(
                "xy_expectation", self.conf.pixel_per_meter
            ),
            "xy_recall_2m": Location2DRecall(2.0, self.conf.pixel_per_meter, "xy_max"),
            "xy_recall_5m": Location2DRecall(5.0, self.conf.pixel_per_meter, "xy_max"),
        }
