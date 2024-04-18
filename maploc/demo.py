# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from perspective2d import PerspectiveFields

from . import logger
from .data.image import pad_image, rectify_image, resize_image
from .evaluation.run import pretrained_models, resolve_checkpoint_path
from .models.orienternet import OrienterNet
from .models.voting import argmax_xyr, fuse_gps
from .osm.raster import Canvas
from .utils.exif import EXIF
from .utils.geo import BoundaryBox, Projection
from .utils.io import read_image
from .utils.wrappers import Camera

try:
    from geopy.geocoders import Nominatim

    geolocator = Nominatim(user_agent="orienternet")
except ImportError:
    geolocator = None


class ImageCalibrator(PerspectiveFields):
    def __init__(self, version: str = "Paramnet-360Cities-edina-centered"):
        super().__init__(version)
        self.eval()

    def run(
        self,
        image_rgb: np.ndarray,
        focal_length: Optional[float] = None,
        exif: Optional[EXIF] = None,
    ) -> Tuple[Tuple[float, float], Camera]:
        h, w, *_ = image_rgb.shape
        if focal_length is None and exif is not None:
            _, focal_ratio = exif.extract_focal()
            if focal_ratio != 0:
                focal_length = focal_ratio * max(h, w)

        calib = self.inference(img_bgr=image_rgb[..., ::-1])
        roll_pitch = (calib["pred_roll"].item(), calib["pred_pitch"].item())
        if focal_length is None:
            vfov = calib["pred_vfov"].item()
            focal_length = h / 2 / np.tan(np.deg2rad(vfov) / 2)

        camera = Camera.from_dict(
            {
                "model": "SIMPLE_PINHOLE",
                "width": w,
                "height": h,
                "params": [focal_length, w / 2 + 0.5, h / 2 + 0.5],
            }
        )
        return roll_pitch, camera


def parse_location_prior(
    exif: EXIF,
    prior_latlon: Optional[Tuple[float, float]] = None,
    prior_address: Optional[str] = None,
) -> np.ndarray:
    latlon = None
    if prior_latlon is not None:
        latlon = prior_latlon
        logger.info("Using prior latlon %s.", prior_latlon)
    elif prior_address is not None:
        if geolocator is None:
            raise ValueError("geocoding unavailable, install geopy.")
        location = geolocator.geocode(prior_address)
        if location is None:
            logger.info("Could not find any location for address '%s.'", prior_address)
        else:
            logger.info("Using prior address '%s'", location.address)
            latlon = (location.latitude, location.longitude)
    if latlon is None:
        geo = exif.extract_geo()
        if geo:
            alt = geo.get("altitude", 0)  # read if available
            latlon = (geo["latitude"], geo["longitude"], alt)
            logger.info("Using prior location from EXIF.")
        else:
            raise ValueError(
                "No location prior given or found in the image EXIF metadata: "
                "maybe provide the name of a street, building or neighborhood?"
            )
    return np.array(latlon)


class Demo:
    def __init__(
        self,
        experiment_or_path: Optional[str] = "OrienterNet_MGL",
        device=None,
        **kwargs
    ):
        if experiment_or_path in pretrained_models:
            experiment_or_path, _ = pretrained_models[experiment_or_path]
        path = resolve_checkpoint_path(experiment_or_path)
        ckpt = torch.load(path, map_location=(lambda storage, loc: storage))
        config = ckpt["hyper_parameters"]
        config.model.update(kwargs)
        config.model.image_encoder.backbone.pretrained = False

        model = OrienterNet(config.model).eval()
        state = {k[len("model.") :]: v for k, v in ckpt["state_dict"].items()}
        model.load_state_dict(state, strict=True)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device)

        self.calibrator = ImageCalibrator().to(device)

        self.config = config
        self.device = device

    def read_input_image(
        self,
        image_path: str,
        prior_latlon: Optional[Tuple[float, float]] = None,
        prior_address: Optional[str] = None,
        focal_length: Optional[float] = None,
        tile_size_meters: int = 64,
    ) -> Tuple[np.ndarray, Camera, Tuple[str, str], Projection, BoundaryBox]:
        image = read_image(image_path)
        with open(image_path, "rb") as fid:
            exif = EXIF(fid, lambda: image.shape[:2])

        gravity, camera = self.calibrator.run(image, focal_length, exif)
        logger.info("Using (roll, pitch) %s.", gravity)

        latlon = parse_location_prior(exif, prior_latlon, prior_address)
        proj = Projection(*latlon)
        center = proj.project(latlon)
        bbox = BoundaryBox(center, center) + tile_size_meters
        return image, camera, gravity, proj, bbox

    def prepare_data(
        self,
        image: np.ndarray,
        camera: Camera,
        canvas: Canvas,
        gravity: Optional[Tuple[float]] = None,
    ) -> Dict[str, torch.Tensor]:
        assert image.shape[:2][::-1] == tuple(camera.size.tolist())
        target_focal_length = self.config.data.resize_image / 2
        factor = target_focal_length / camera.f
        size = (camera.size * factor).round().int()

        image = torch.from_numpy(image).permute(2, 0, 1).float().div_(255)
        valid = None
        if gravity is not None:
            roll, pitch = gravity
            image, valid = rectify_image(
                image,
                camera.float(),
                roll=-roll,
                pitch=-pitch,
            )
        image, _, camera, *maybe_valid = resize_image(
            image, size.tolist(), camera=camera, valid=valid
        )
        valid = None if valid is None else maybe_valid

        max_stride = max(self.model.image_encoder.layer_strides)
        size = (torch.ceil(size / max_stride) * max_stride).int()
        image, valid, camera = pad_image(
            image, size.tolist(), camera, crop_and_center=True
        )

        return {
            "image": image,
            "map": torch.from_numpy(canvas.raster).long(),
            "camera": camera.float(),
            "valid": valid,
        }

    def localize(self, image: np.ndarray, camera: Camera, canvas: Canvas, **kwargs):
        data = self.prepare_data(image, camera, canvas, **kwargs)
        data_ = {k: v.to(self.device)[None] for k, v in data.items()}
        with torch.no_grad():
            pred = self.model(data_)

        xy_gps = canvas.bbox.center
        uv_gps = torch.from_numpy(canvas.to_uv(xy_gps))

        lp_xyr = pred["log_probs"].squeeze(0)
        tile_size = canvas.bbox.size.min() / 2
        sigma = tile_size - 20  # 20 meters margin
        lp_xyr = fuse_gps(
            lp_xyr,
            uv_gps.to(lp_xyr),
            self.config.model.pixel_per_meter,
            sigma=sigma,
        )
        xyr = argmax_xyr(lp_xyr).cpu()

        prob = lp_xyr.exp().cpu()
        neural_map = pred["map"]["map_features"][0].squeeze(0).cpu()
        return xyr[:2], xyr[2], prob, neural_map, data["image"]
