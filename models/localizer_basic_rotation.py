# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import copy

import omegaconf  # @manual
import torch
from torch.nn.functional import normalize, pad

from .base import BaseModel
from .feature_extractor import FeatureExtractor
from .hough_voting import argmax_xyr, expectation_xyr, log_softmax_spatial, nll_loss_xyr
from .map_encoder import MapEncoder
from .metrics import AngleError, Location2DError
from .utils import GlobalPooling


def compute_rotation_features(features, num_basis, num_rotations):
    b, d, h, w = features.shape
    assert (d % num_basis) == 0

    features = features.refine_names("B", "D", "H", "W")
    basis = features.unflatten("D", [("D", d // num_basis), ("N", num_basis)])
    basis = basis.align_to("B", "D", "H", "W", "N").rename(None)
    basis_padded = pad(basis, [0, 1, 0, 0, 0, 0], mode="circular")
    basis_padded = basis_padded.unsqueeze(-1).refine_names("B", "D", "H", "W", "N", "K")

    w = torch.arange(0, 1, num_basis / num_rotations).to(features).refine_names("K")
    interp = (1 - w) * basis_padded[..., :-1, :] + w * basis_padded[..., 1:, :]
    features_xyr = interp.flatten(("N", "K"), "N")
    return features_xyr


class BasicLocalizer(BaseModel):
    default_conf = {
        "image_encoder": "???",
        "map_encoder": "???",
        "latent_dim": "???",
        "pixel_per_meter": 2,
        "num_basis": 4,
        "num_rotations": 32,
    }

    def _init(self, conf):
        self.image_encoder = FeatureExtractor(conf.image_encoder.backbone)
        self.image_pooling = GlobalPooling(kind=conf.image_encoder.pooling)

        map_conf = copy.deepcopy(conf.map_encoder)
        with omegaconf.read_write(map_conf):
            with omegaconf.open_dict(map_conf):
                map_conf.backbone.output_dim = conf.latent_dim * conf.num_basis
        self.map_encoder = MapEncoder(map_conf)

        scale = torch.nn.Parameter(torch.tensor(1.0))
        self.register_parameter("feature_scale", scale)

    def _forward(self, data):
        image_features = self.image_encoder(data)["feature_maps"]
        image_descriptor = self.image_pooling(image_features[0])
        map_pred = self.map_encoder(data)

        level = 0
        map_features = map_pred["map_features"][level]
        features_xyr = compute_rotation_features(
            map_features, self.conf.num_basis, self.conf.num_rotations
        ).rename(None)

        features_xyr = normalize(features_xyr, dim=1) * self.feature_scale
        image_descriptor = normalize(image_descriptor, dim=1) * self.feature_scale

        scores = torch.einsum("bd,bdhwn->bhwn", image_descriptor, features_xyr)
        if "log_prior" in map_pred:
            scores += map_pred["log_prior"][level].unsqueeze(-1)
        log_probs = log_softmax_spatial(scores)
        xyr_max = argmax_xyr(scores).to(scores)
        xyr_avg, _ = expectation_xyr(log_probs.exp())

        return {
            "scores": scores,
            "log_probs": log_probs,
            "xyr_max": xyr_max,
            "xy_max": xyr_max[..., :2],
            "yaw_max": xyr_max[..., 2],
            "xyr_expectation": xyr_avg,
            "xy_expectation": xyr_avg[..., :2],
            "yaw_expectation": xyr_avg[..., 2],
            "image_features": image_features,
            "image_descriptor": image_descriptor,
            "map": map_pred,
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
