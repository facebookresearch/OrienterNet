# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import numpy as np
import torch
from torch.nn.functional import grid_sample, normalize

from .base import BaseModel
from .feature_extractor import FeatureExtractor
from .hough_voting import (
    argmax_xyr,
    expectation_xyr,
    log_softmax_spatial,
    nll_loss_xyr,
    SparseMapSampler,
)
from .map_encoder import MapEncoder
from .metrics import AngleError, Location2DError


class BEVLocalizer(BaseModel):
    default_conf = {
        "image_encoder": "???",
        "map_encoder": "???",
        "polar_projection": "polar_projection",
        "bev_net": "???",
        "latent_dim": "???",
        "matching_dim": "???",
        "z_max": "???",
        "x_max": "???",
        "pixel_per_meter": "???",
        "num_rotations": "???",
        "normalize_features": False,
    }

    def _init(self, conf):
        self.image_encoder = FeatureExtractor(conf.image_encoder.backbone)
        self.map_encoder = MapEncoder(conf.map_encoder)
        # self.bev_net = BEVNet(conf.bev_net)
        self.map_sampler = SparseMapSampler(conf.num_rotations)

    def _forward(self, data):
        map_pred = self.map_encoder(data)
        features_map = map_pred["map_features"][0]

        level = 0
        features_image = self.image_encoder(data)["feature_maps"][level]
        camera = data["camera"].scale(1 / self.image_encoder.scales[level])

        p3d = data["pointcloud"]
        p3d_cam = p3d @ data["R_rect2cam"].transpose(-1, -2)
        p2d, valid_projection = camera.world2image(p3d_cam)

        size = p2d.new_tensor(features_image.shape[-2:][::-1])
        p2d_norm = (p2d + 0.5) / size * 2 - 1
        features_p3d = grid_sample(
            features_image, p2d_norm.unsqueeze(-2), align_corners=False
        ).squeeze(-1)
        # TODO: sample any confidence map? or add as last feature channel + split

        p2d_bev = p3d[..., :2] * p3d.new_tensor([1, -1]) * self.conf.pixel_per_meter

        # optionally: discretize to a grid, pool, and run a dense or sparse CNN
        # bev_pred = self.bev_net({"input": features_bev_proj})
        # features_bev = bev_pred["output"]
        features_bev = features_p3d
        # TODO: predict a BEV confidence?

        if self.conf.normalize_features:
            features_bev = normalize(features_bev, dim=1)
            map_pred["map_features"][0] = features_map = normalize(features_map, dim=1)

        features_map_sparse, valid_map_sparse = self.map_sampler(features_map, p2d_bev)
        valid_points = valid_projection[..., None, None, None] & valid_map_sparse
        # We sum the scores over all points.
        # Those that are invalid should have zero features because of the zero padding
        scores = torch.einsum("bdnhwk,bdn->bhwk", features_map_sparse, features_bev)
        scores /= np.sqrt(features_map.shape[1])  # preserve the feature variance
        scores /= valid_points.to(scores).sum(-4).clamp(min=1.0)
        scores_matching = scores

        if "log_prior" in map_pred:
            scores = scores + map_pred["log_prior"][level].unsqueeze(-1)
        log_probs = log_softmax_spatial(scores)
        xyr_max = argmax_xyr(scores).to(scores)
        xyr_avg, _ = expectation_xyr(log_probs.exp())

        return {
            "scores": scores,
            "scores_matching": scores_matching,
            "log_probs": log_probs,
            "xyr_max": xyr_max,
            "xy_max": xyr_max[..., :2],
            "yaw_max": xyr_max[..., 2],
            "xyr_expectation": xyr_avg,
            "xy_expectation": xyr_avg[..., :2],
            "yaw_expectation": xyr_avg[..., 2],
            "features_image": features_image,
            "features_p3d": features_p3d,
            "features_bev": features_bev,
            "valid_points": valid_points,
            "map": map_pred,
            # "bev": bev_pred,
        }

    def loss(self, pred, data):
        nll = nll_loss_xyr(
            pred["log_probs"], data["xy"], data["roll_pitch_yaw"][..., -1]
        )
        return {"total": nll, "nll": nll}

    def metrics(self):
        return {
            "xy_max_error": Location2DError("xy_max", self.conf.pixel_per_meter),
            "xy_expectation_error": Location2DError(
                "xy_expectation", self.conf.pixel_per_meter
            ),
            "yaw_max_error": AngleError("yaw_max"),
        }
