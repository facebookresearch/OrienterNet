# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import numpy as np
from torch.nn.functional import normalize

from . import get_model
from .base import BaseModel
from .bev_net import BEVNet
from .bev_projection import CartesianProjection, PolarProjection, PolarProjectionPlane
from .feature_extractor import FeatureExtractor
from .hough_voting import (
    argmax_xyr,
    conv2d_fft_batchwise,
    expectation_xyr,
    log_softmax_spatial,
    nll_loss_xyr,
    TemplateSampler,
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
        self.bev_net = BEVNet(conf.bev_net)

        ppm = conf.pixel_per_meter
        if conf.polar_projection == "polar_projection":
            self.projection_polar = PolarProjection(conf.z_max, ppm)
        elif conf.polar_projection == "polar_projection_plane":
            self.projection_polar = PolarProjectionPlane(conf.z_max, ppm)
        else:
            self.projection_polar = get_model(conf.polar_projection.name)(
                conf.polar_projection
            )
        self.projection_bev = CartesianProjection(conf.z_max, conf.x_max, ppm)
        self.template_sampler = TemplateSampler(
            self.projection_bev.grid_xz, ppm, conf.num_rotations
        )

    def _forward(self, data):
        map_pred = self.map_encoder(data)
        features_map = map_pred["map_features"][0]

        level = 0
        features_image = self.image_encoder(data)["feature_maps"][level]
        camera = data["camera"].scale(1 / self.image_encoder.scales[level])

        if isinstance(self.projection_polar, PolarProjectionPlane):
            features_polar, valid_polar, _ = self.projection_polar(
                features_image, data["ground_plane"], camera
            )
        elif isinstance(self.projection_polar, PolarProjection):
            features_polar, valid_polar, _ = self.projection_polar(
                features_image, data["camera_height"], camera
            )
        else:
            pred_polar = self.projection_polar(
                {**data, "camera": camera, "image_features": features_image}
            )
            features_polar = pred_polar["polar_features"]
            valid_polar = pred_polar["valid"]
        features_bev_proj, valid_bev, _ = self.projection_bev(
            features_polar, valid_polar, camera
        )
        bev_pred = self.bev_net({"input": features_bev_proj})
        features_bev = bev_pred["output"]

        if self.conf.normalize_features:
            features_bev = normalize(features_bev, dim=1)
            map_pred["map_features"][0] = features_map = normalize(features_map, dim=1)

        template = features_bev
        if "confidence" in bev_pred:
            template = template * bev_pred["confidence"].unsqueeze(1)
        template = template.masked_fill(~valid_bev.unsqueeze(1), 0.0)
        templates = self.template_sampler(template)
        scores = conv2d_fft_batchwise(features_map, templates)
        scores = scores / np.sqrt(features_map.shape[1])  # number of channels

        # Reweight the different rotations based on the number of valid pixels
        # in each template. Axis-aligned rotation have the maximum number of valid pxiels.
        valid_templates = self.template_sampler(valid_bev.float()[None]) > (1 - 1e-4)
        num_valid = valid_templates.float().sum((-3, -2, -1))
        reweight = num_valid.max(-1).values.unsqueeze(-1) / num_valid
        scores = scores * reweight[..., None, None]

        scores = scores_matching = scores.permute(0, 2, 3, 1)  # B x H x W x N
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
            "features_polar": features_polar,
            "features_bev_proj": features_bev_proj,
            "features_bev": features_bev,
            "valid_bev": valid_bev.squeeze(1),
            "map": map_pred,
            "bev": bev_pred,
            "templates": templates,
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
