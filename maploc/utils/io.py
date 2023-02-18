# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import json

import cv2
import numpy as np


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


def format_json(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.int64):
        return int(x)
    if isinstance(x, dict):
        return {k: format_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return list(map(format_json, x))
    return x


def write_json(path, data):
    with open(path, "w") as f:
        json.dump(format_json(data), f)
