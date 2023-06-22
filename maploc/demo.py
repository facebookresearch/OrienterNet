# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import Optional, Tuple

import torch
import numpy as np

from . import logger
from .evaluation.run import resolve_checkpoint_path, pretrained_models
from .models.orienternet import OrienterNet
from .models.voting import fuse_gps, argmax_xyr
from .data.image import resize_image, pad_image, rectify_image
from .osm.raster import Canvas
from .utils.wrappers import Camera
from .utils.io import read_image
from .utils.geo import BoundaryBox, Projection
from .utils.exif import EXIF

try:
    from geopy.geocoders import Nominatim

    geolocator = Nominatim(user_agent="orienternet")
except ImportError:
    geolocator = None

try:
    from gradio_client import Client

    calibrator = Client("https://jinlinyi-perspectivefields.hf.space/")
except (ImportError, ValueError):
    calibrator = None


def image_calibration(image_path):
    logger.info("Calling the PerspectiveFields calibrator, this may take some time.")
    result = calibrator.predict(
        image_path, "NEW:Paramnet-360Cities-edina-centered", api_name="/predict"
    )
    result = dict(r.rsplit(" ", 1) for r in result[1].split("\n"))
    roll_pitch = float(result["roll"]), float(result["pitch"])
    return roll_pitch, float(result["vertical fov"])


def camera_from_exif(exif: EXIF, fov: Optional[float] = None) -> Camera:
    w, h = image_size = exif.extract_image_size()
    _, f_ratio = exif.extract_focal()
    if f_ratio == 0:
        if fov is not None:
            # This is the vertical FoV.
            f = h / 2 / np.tan(np.deg2rad(fov) / 2)
        else:
            return None
    else:
        f = f_ratio * max(image_size)
    return Camera.from_dict(
        dict(
            model="SIMPLE_PINHOLE",
            width=w,
            height=h,
            params=[f, w / 2 + 0.5, h / 2 + 0.5],
        )
    )


def read_input_image(
    image_path: str,
    prior_latlon: Optional[Tuple[float, float]] = None,
    prior_address: Optional[str] = None,
    fov: Optional[float] = None,
    tile_size_meters: int = 64,
):
    image = read_image(image_path)

    roll_pitch = None
    if calibrator is not None:
        roll_pitch, fov = image_calibration(image_path)
    else:
        logger.info("Could not call PerspectiveFields, maybe install gradio_client?")
    if roll_pitch is not None:
        logger.info("Using (roll, pitch) %s.", roll_pitch)

    with open(image_path, "rb") as fid:
        exif = EXIF(fid, lambda: image.shape[:2])
    camera = camera_from_exif(exif, fov)
    if camera is None:
        raise ValueError(
            "No camera intrinsics found in the EXIF, provide an FoV guess."
        )

    latlon = None
    if prior_latlon is not None:
        latlon = prior_latlon
        logger.info("Using prior latlon %s.", prior_latlon)
    if prior_address is not None:
        if geolocator is None:
            raise ValueError("geocoding unavailable, install geopy.")
        location = geolocator.geocode(prior_address)
        if location is None:
            logger.info("Could not find any location for %s.", prior_address)
        else:
            logger.info("Using prior address: %s", location.address)
            latlon = (location.latitude, location.longitude)
    if latlon is None:
        geo = exif.extract_geo()
        if geo:
            alt = geo.get("altitude", 0)  # read if available
            latlon = (geo["latitude"], geo["longitude"], alt)
            logger.info("Using prior location from EXIF.")
        else:
            logger.info("Could not find any prior location in EXIF.")
    if latlon is None:
        raise ValueError("Need prior latlon")
    latlon = np.array(latlon)

    proj = Projection(*latlon)
    center = proj.project(latlon)
    bbox = BoundaryBox(center, center) + tile_size_meters
    return image, camera, roll_pitch, proj, bbox, latlon


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
        model = model.to(device)

        self.model = model
        self.config = config
        self.device = device

    def prepare_data(
        self,
        image: np.ndarray,
        camera: Camera,
        canvas: Canvas,
        roll_pitch: Optional[Tuple[float]] = None,
    ):
        assert image.shape[:2][::-1] == tuple(camera.size.tolist())
        target_focal_length = self.config.data.resize_image / 2
        factor = target_focal_length / camera.f
        size = (camera.size * factor).round().int()

        image = torch.from_numpy(image).permute(2, 0, 1).float().div_(255)
        valid = None
        if roll_pitch is not None:
            roll, pitch = roll_pitch
            image, valid = rectify_image(
                image,
                camera.float(),
                roll=-roll,
                pitch=-pitch,
            )
        image, _, camera, *maybe_valid = resize_image(
            image, size.numpy(), camera=camera, valid=valid
        )
        valid = None if valid is None else maybe_valid

        max_stride = max(self.model.image_encoder.layer_strides)
        size = (np.ceil((size / max_stride)) * max_stride).int()
        image, valid, camera = pad_image(
            image, size.numpy(), camera, crop_and_center=True
        )

        return dict(
            image=image,
            map=torch.from_numpy(canvas.raster).long(),
            camera=camera.float(),
            valid=valid,
        )

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
        return xyr[:2], xyr[1], prob, neural_map, data["image"]
