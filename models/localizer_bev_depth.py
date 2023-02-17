# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import numpy as np
import torch
from torch.nn.functional import normalize, softmax

from . import get_model
from .base import BaseModel
from .bev_net import BEVNet
from .bev_projection import CartesianProjection, PolarProjectionDepth
from .bev_transformer import fourier_positional_encoding
from .hough_voting import (
    argmax_xyr,
    conv2d_fft_batchwise,
    expectation_xyr,
    log_softmax_spatial,
    nll_loss_xyr,
    TemplateSampler,
)
from .map_encoder import MapEncoder
from .metrics import AngleError, AngleRecall, Location2DError, Location2DRecall
from .utils import GlobalPooling


class BEVLocalizer(BaseModel):
    default_conf = {
        "image_encoder": "???",
        "map_encoder": "???",
        "bev_net": "???",
        "latent_dim": "???",
        "matching_dim": "???",
        "do_joint_retrieval": False,
        "retrieval_dim": 128,
        "depth_parameterization": "scale",
        "scale_range": [0, 8],
        "num_scale_bins": "???",
        "norm_depth_scores": True,
        "z_min": None,
        "z_max": "???",
        "x_max": "???",
        "pixel_per_meter": "???",
        "num_rotations": "???",
        "normalize_features": False,
        "normalize_scores_by_dim": True,
        "normalize_scores_by_num_valid": False,
        "padding_matching": "constant",
        "apply_map_prior": True,
        "prior_renorm": True,
    }

    def _init(self, conf):
        Encoder = get_model(conf.image_encoder.get("name", "feature_extractor"))
        self.image_encoder = Encoder(conf.image_encoder.backbone)
        map_dim = conf.map_encoder.get(
            "output_dim", conf.map_encoder.backbone.get("output_dim")
        )
        if conf.do_joint_retrieval:
            map_dim += conf.retrieval_dim
        self.map_encoder = MapEncoder({**conf.map_encoder, "output_dim": map_dim})
        self.bev_net = None if conf.bev_net is None else BEVNet(conf.bev_net)
        if conf.do_joint_retrieval:
            self.retrieval_head = torch.nn.Sequential(
                GlobalPooling("max"),
                torch.nn.Linear(
                    self.image_encoder.skip_dims[-1], conf.retrieval_dim * 2
                ),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(conf.retrieval_dim * 2, conf.retrieval_dim),
            )

        ppm = conf.pixel_per_meter
        self.projection_polar = PolarProjectionDepth(
            conf.z_max,
            ppm,
            conf.scale_range,
            conf.z_min,
            log_prob=not self.conf.norm_depth_scores,
        )
        self.projection_bev = CartesianProjection(
            conf.z_max, conf.x_max, ppm, conf.z_min
        )
        self.template_sampler = TemplateSampler(
            self.projection_bev.grid_xz, ppm, conf.num_rotations
        )

        if conf.depth_parameterization == "scale":
            # add a dustbin when the scores are normalized
            scale_bins = conf.num_scale_bins + int(conf.norm_depth_scores)
            self.scale_classifier = torch.nn.Linear(conf.latent_dim, scale_bins)
        elif conf.depth_parameterization == "bev_depth":
            depth_bins = len(self.projection_polar.depth_steps) + int(
                conf.norm_depth_scores
            )
            self.depth_classifier = torch.nn.Linear(conf.latent_dim, depth_bins)
        elif conf.depth_parameterization == "bev_depth_posenc":
            assert not conf.norm_depth_scores
            self.depth_classifier = torch.nn.Linear(conf.latent_dim, conf.latent_dim)
            self.posenc_projection = torch.nn.Linear(conf.latent_dim, conf.latent_dim)
            steps = torch.arange(
                len(self.projection_polar.depth_steps), dtype=torch.float
            )
            posenc = fourier_positional_encoding(steps, conf.latent_dim, 4 * len(steps))
            self.register_buffer("posenc", posenc)
        elif conf.depth_parameterization == "bev_depth_log":
            assert not conf.norm_depth_scores
            self.log_depth_classifier = torch.nn.Linear(
                conf.latent_dim, conf.num_scale_bins
            )
        else:
            raise ValueError(
                f"Unknown depth parameterization: {conf.depth_parameterization}"
            )
        if conf.bev_net is None:
            self.feature_projection = torch.nn.Linear(
                conf.latent_dim, conf.matching_dim
            )

    def _forward(self, data):
        pred = {}
        map_pred = pred["map"] = self.map_encoder(data)
        features_map = map_pred["map_features"][0]

        level = 0
        image_pred = pred["image"] = self.image_encoder(data)
        features_image = image_pred["feature_maps"][level]
        camera = data["camera"].scale(1 / self.image_encoder.scales[level])

        if self.conf.do_joint_retrieval:
            features_map, map_retrieval = features_map.split(
                [self.conf.matching_dim, self.conf.retrieval_dim], 1
            )
            features_global = self.retrieval_head(image_pred["skip_features"][-1])
            pred["scores_retrieval"] = torch.einsum(
                "bd,bdhw->bhw", features_global, map_retrieval
            )

        features = features_image
        if self.conf.bev_net is None:
            features = self.feature_projection(features.permute(0, 2, 3, 1)).permute(
                0, 3, 1, 2
            )

        if self.conf.depth_parameterization == "scale":
            scales = self.scale_classifier(features_image.permute(0, 2, 3, 1))
            if self.conf.norm_depth_scores:
                scales = softmax(scales, dim=-1)[..., :-1]  # remove the dustbin
            pred["pixel_scales"] = scales
            features_polar, mask_polar = self.projection_polar(features, scales, camera)
        elif self.conf.depth_parameterization == "bev_depth":
            depths = self.depth_classifier(features_image.permute(0, 2, 3, 1))
            if self.conf.norm_depth_scores:
                depths = softmax(depths, dim=-1)[..., :-1]
            pred["pixel_depths"] = depths
            features_polar, mask_polar = self.projection_polar(
                features, polar_depths=depths
            )
        elif self.conf.depth_parameterization == "bev_depth_posenc":
            keys = self.depth_classifier(features_image.permute(0, 2, 3, 1))
            queries = self.posenc_projection(self.posenc)
            depths = (
                torch.einsum("...hwd,nd->...hwn", keys, queries)
                / self.conf.latent_dim**0.5
            )
            pred["pixel_depths"] = depths
            features_polar, mask_polar = self.projection_polar(
                features, polar_depths=depths
            )
        elif self.conf.depth_parameterization == "bev_depth_log":
            log_depths = self.log_depth_classifier(features_image.permute(0, 2, 3, 1))
            pred["pixel_log_depths"] = log_depths
            features_polar, mask_polar = self.projection_polar(
                features, polar_log_depths=log_depths
            )
        else:
            raise ValueError(self.conf.depth_parameterization)

        # TODO: how to use the mask?
        valid_polar = None  # all pixels are valid?
        features_bev, valid_bev, _ = self.projection_bev(
            features_polar, valid_polar, camera
        )
        bev_pred = {}
        if self.conf.bev_net is not None:
            bev_pred = pred["bev"] = self.bev_net({"input": features_bev})
            features_bev = bev_pred["output"]

        if self.conf.normalize_features:
            features_bev = normalize(features_bev, dim=1)
            map_pred["map_features"][0] = features_map = normalize(features_map, dim=1)

        template = features_bev
        if "confidence" in bev_pred:
            template = template * bev_pred["confidence"].unsqueeze(1)
        template = template.masked_fill(~valid_bev.unsqueeze(1), 0.0)
        templates = self.template_sampler(template)
        scores = conv2d_fft_batchwise(
            features_map, templates, padding_mode=self.conf["padding_matching"]
        )
        if self.conf.normalize_scores_by_dim:
            scores = scores / np.sqrt(features_map.shape[1])  # number of channels

        # Reweight the different rotations based on the number of valid pixels
        # in each template. Axis-aligned rotation have the maximum number of valid pixels.
        valid_templates = self.template_sampler(valid_bev.float()[None]) > (1 - 1e-4)
        num_valid = valid_templates.float().sum((-3, -2, -1))
        if self.conf.normalize_scores_by_num_valid:
            reweight = 1.0 / num_valid
        else:
            reweight = num_valid.max(-1).values.unsqueeze(-1) / num_valid
        scores = scores * reweight[..., None, None]

        scores = pred["scores_matching"] = scores.permute(0, 2, 3, 1)  # B x H x W x N
        add_prior = "log_prior" in map_pred and self.conf.apply_map_prior
        # legacy: log_prior is already sigmoid-normalized, so this is weird
        if add_prior and self.conf.prior_renorm:
            scores = scores + map_pred["log_prior"][0].unsqueeze(-1)
        if "scores_retrieval" in pred:
            scores = scores + pred["scores_retrieval"].unsqueeze(-1)
        if "map_mask" in data:
            scores.masked_fill_(~data["map_mask"][..., None], -np.inf)
        log_probs = log_softmax_spatial(scores)
        # the better way to add the prior: after softmax normalization
        if add_prior and not self.conf.prior_renorm:
            log_probs = log_probs + map_pred["log_prior"][0].unsqueeze(-1)
        with torch.no_grad():
            xyr_max = argmax_xyr(scores).to(scores)
            xyr_avg, _ = expectation_xyr(log_probs.exp())

        return {
            **pred,
            "scores": scores,
            "log_probs": log_probs,
            "xyr_max": xyr_max,
            "xy_max": xyr_max[..., :2],
            "yaw_max": xyr_max[..., 2],
            "xyr_expectation": xyr_avg,
            "xy_expectation": xyr_avg[..., :2],
            "yaw_expectation": xyr_avg[..., 2],
            "features_image": features_image,
            "features_polar": features_polar,
            "features_bev": features_bev,
            "mask_polar": mask_polar,
            "valid_bev": valid_bev.squeeze(1),
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
            "xy_recall_2m": Location2DRecall(2.0, self.conf.pixel_per_meter, "xy_max"),
            "xy_recall_5m": Location2DRecall(5.0, self.conf.pixel_per_meter, "xy_max"),
            "yaw_recall_2°": AngleRecall(2.0, "yaw_max"),
            "yaw_recall_5°": AngleRecall(5.0, "yaw_max"),
        }
