# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
from pathlib import Path

import numpy as np
import httpx
import asyncio
from aiolimiter import AsyncLimiter
import tqdm

from opensfm.pygeometry import Camera, Pose
from opensfm.pymap import Shot

from ... import logger
from ...utils.geo import Projection


semaphore = asyncio.Semaphore(100)  # number of parallel threads.
image_filename = "{image_id}.jpg"
info_filename = "{image_id}.json"


class MapillaryDownloader:
    image_fields = (
        "id",
        "height",
        "width",
        "camera_parameters",
        "camera_type",
        "captured_at",
        "compass_angle",
        "geometry",
        "altitude",
        "computed_compass_angle",
        "computed_geometry",
        "computed_altitude",
        "computed_rotation",
        "thumb_2048_url",
        "thumb_original_url",
        "sequence",
        "sfm_cluster",
    )
    image_info_url = (
        "https://graph.mapillary.com/{image_id}?access_token={token}&fields={fields}"
    )
    seq_info_url = "https://graph.mapillary.com/image_ids?access_token={token}&sequence_id={seq_id}"
    max_requests_per_minute = 50_000

    def __init__(self, token: str):
        self.token = token
        self.client = httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(retries=20), timeout=20.0
        )
        self.limiter = AsyncLimiter(self.max_requests_per_minute // 2, time_period=60)

    async def call_api(self, url: str):
        async with self.limiter:
            r = await self.client.get(url)
        if not r.is_success:
            logger.error("Error in API call: %s", r.text)
        return r

    async def get_image_info(self, image_id: int):
        url = self.image_info_url.format(
            image_id=image_id,
            token=self.token,
            fields=",".join(self.image_fields),
        )
        r = await self.call_api(url)
        if r.is_success:
            return json.loads(r.text)

    async def get_sequence_info(self, seq_id: str):
        url = self.seq_info_url.format(seq_id=seq_id, token=self.token)
        r = await self.call_api(url)
        if r.is_success:
            return json.loads(r.text)

    async def download_image_pixels(self, url: str, path: Path):
        r = await self.call_api(url)
        if r.is_success:
            with open(path, "wb") as fid:
                fid.write(r.content)
        return r.is_success

    async def get_image_info_cached(self, image_id: int, path: Path):
        if path.exists():
            info = json.loads(path.read_text())
        else:
            info = await self.get_image_info(image_id)
            path.write_text(json.dumps(info))
        return info

    async def download_image_pixels_cached(self, url: str, path: Path):
        if path.exists():
            return True
        else:
            return await self.download_image_pixels(url, path)


async def fetch_images_in_sequence(i, downloader):
    async with semaphore:
        info = await downloader.get_sequence_info(i)
    image_ids = [int(d["id"]) for d in info["data"]]
    return i, image_ids


async def fetch_images_in_sequences(sequence_ids, downloader):
    seq_to_images_ids = {}
    tasks = [fetch_images_in_sequence(i, downloader) for i in sequence_ids]
    for task in tqdm.asyncio.tqdm.as_completed(tasks):
        i, image_ids = await task
        seq_to_images_ids[i] = image_ids
    return seq_to_images_ids


async def fetch_image_info(i, downloader, dir_):
    async with semaphore:
        path = dir_ / info_filename.format(image_id=i)
        info = await downloader.get_image_info_cached(i, path)
    return i, info


async def fetch_image_infos(image_ids, downloader, dir_):
    infos = {}
    num_fail = 0
    tasks = [fetch_image_info(i, downloader, dir_) for i in image_ids]
    for task in tqdm.asyncio.tqdm.as_completed(tasks):
        i, info = await task
        if info is None:
            num_fail += 1
        else:
            infos[i] = info
    return infos, num_fail


async def fetch_image_pixels(i, url, downloader, dir_, overwrite=False):
    async with semaphore:
        path = dir_ / image_filename.format(image_id=i)
        if overwrite:
            path.unlink(missing_ok=True)
        success = await downloader.download_image_pixels_cached(url, path)
    return i, success


async def fetch_images_pixels(image_urls, downloader, dir_):
    num_fail = 0
    tasks = [fetch_image_pixels(*id_url, downloader, dir_) for id_url in image_urls]
    for task in tqdm.asyncio.tqdm.as_completed(tasks):
        i, success = await task
        num_fail += not success
    return num_fail


def opensfm_camera_from_info(info: dict) -> Camera:
    cam_type = info["camera_type"]
    if cam_type == "perspective":
        camera = Camera.create_perspective(*info["camera_parameters"])
    elif cam_type == "fisheye":
        camera = Camera.create_fisheye(*info["camera_parameters"])
    elif Camera.is_panorama(cam_type):
        camera = Camera.create_spherical()
    else:
        raise ValueError(cam_type)
    camera.width = info["width"]
    camera.height = info["height"]
    camera.id = info["id"]
    return camera


def opensfm_shot_from_info(info: dict, projection: Projection) -> Shot:
    latlong = info["computed_geometry"]["coordinates"][::-1]
    alt = info["computed_altitude"]
    xyz = projection.project(np.array([*latlong, alt]), return_z=True)
    c_rotvec_w = np.array(info["computed_rotation"])
    pose = Pose()
    pose.set_from_cam_to_world(-c_rotvec_w, xyz)
    camera = opensfm_camera_from_info(info)
    return latlong, Shot(info["id"], camera, pose)
