# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import requests
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

from .. import logger

DATA_URL = "https://cvg-data.inf.ethz.ch/OrienterNet_CVPR2023"


def read_image(path, grayscale=False):
    if grayscale:
        mode = cv2.IMREAD_GRAYSCALE
    else:
        mode = cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise ValueError(f"Cannot read image {path}.")
    if not grayscale and len(image.shape) == 3:
        image = np.ascontiguousarray(image[:, :, ::-1])  # BGR to RGB
    return image


def write_torch_image(path, image):
    image_cv2 = np.round(image.clip(0, 1) * 255).astype(int)[..., ::-1]
    cv2.imwrite(str(path), image_cv2)


class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.ndarray, torch.Tensor)):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        return json.JSONEncoder.default(self, obj)


def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, cls=JSONEncoder)


def download_file(url, path):
    path = Path(path)
    if path.is_dir():
        path = path / Path(url).name
    path.parent.mkdir(exist_ok=True, parents=True)
    logger.info("Downloading %s to %s.", url, path)
    with requests.get(url, stream=True) as r:
        total_length = int(r.headers.get("Content-Length"))
        with tqdm.wrapattr(r.raw, "read", total=total_length, desc="") as raw:
            with open(path, "wb") as output:
                shutil.copyfileobj(raw, output)
    return path
