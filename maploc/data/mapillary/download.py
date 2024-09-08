# Copyright (c) Meta Platforms, Inc. and affiliates.

import asyncio
import json
from functools import partial
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import httpx
import numpy as np
import tqdm
from aiolimiter import AsyncLimiter
from opensfm.pygeometry import Camera, Pose
from opensfm.pymap import Shot

from ... import logger
from ...utils.geo import BoundaryBox, Projection

semaphore = asyncio.Semaphore(100)  # number of parallel threads.
image_filename = "{image_id}.jpg"
info_filename = "{image_id}.json"


def retry(times, exceptions):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            attempt = 0
            while attempt < times:
                try:
                    return await func(*args, **kwargs)
                except exceptions:
                    attempt += 1
            return await func(*args, **kwargs)

        return wrapper

    return decorator


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
    base_url = "https://graph.mapillary.com"
    image_info_url = "{base}/{image_id}?access_token={token}&fields={fields}"
    image_list_url = (
        "{base}/images?access_token={token}&bbox={bbox}&limit={limit}&fields=id"
    )
    max_requests_per_minute = 50_000
    max_num_results = 5_000  # maximum allowed by mapillary.com

    def __init__(self, token: str):
        self.token = token
        self.client = httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(retries=20), timeout=120.0
        )
        self.limiter = AsyncLimiter(self.max_requests_per_minute // 2, time_period=60)

    @retry(times=5, exceptions=(httpx.RemoteProtocolError, httpx.ReadError))
    async def call_api(self, url: str):
        async with semaphore:
            async with self.limiter:
                r = await self.client.get(url)
        if not r.is_success:
            logger.error("Error in API call: %s, retrying...", r.text)
            raise httpx.ReadError(r.text)
        return r

    async def get_image_list(
        self, bbox: BoundaryBox, is_pano: Optional[bool] = None, **filters
    ):
        # API bbox format: left, bottom, right, top (or minLon, minLat, maxLon, maxLat)
        bbox = ",".join(map(str, (*bbox.min_[::-1], *bbox.max_[::-1])))
        url = self.image_list_url.format(
            base=self.base_url, token=self.token, bbox=bbox, limit=self.max_num_results
        )
        if is_pano is not None:
            url += "&is_pano=" + ("true" if is_pano else "false")
        for name, val in filters.items():
            url += f"&{name}={val}"
        r = await self.call_api(url)
        if r.is_success:
            info = json.loads(r.text)
            image_ids = [int(d["id"]) for d in info["data"]]
            return image_ids
            # return json.loads(r.text)

    async def get_image_info(
        self, image_id: int, fields: Optional[Sequence[str]] = None
    ):
        url = self.image_info_url.format(
            base=self.base_url,
            image_id=image_id,
            token=self.token,
            fields=",".join(fields or self.image_fields),
        )
        r = await self.call_api(url)
        if r.is_success:
            return json.loads(r.text)

    async def get_image_info_cached(
        self, image_id: int, dir_: Optional[Path] = None, **kwargs
    ):
        if dir_ is None:
            return await self.get_image_info(image_id, **kwargs)
        path = dir_ / info_filename.format(image_id=image_id)
        if path.exists():
            info = json.loads(path.read_text())
        else:
            info = await self.get_image_info(image_id, **kwargs)
            if info is not None:
                path.write_text(json.dumps(info))
        return info

    async def download_image_pixels(self, url: str, path: Path):
        r = await self.call_api(url)
        if r.is_success:
            with open(path, "wb") as fid:
                fid.write(r.content)
        return r.is_success

    async def download_image_pixels_cached(self, url: str, path: Path):
        if path.exists():
            return True
        else:
            return await self.download_image_pixels(url, path)


async def _return_with_arg(item, fn):
    ret = await fn(item)
    return item, ret


async def fetch_many(items, fn):
    results = []
    tasks = [_return_with_arg(item, fn) for item in items]
    for task in tqdm.asyncio.tqdm.as_completed(tasks):
        results.append(await task)
    return results


async def fetch_image_infos(image_ids, downloader, **kwargs):
    infos = await fetch_many(
        image_ids, partial(downloader.get_image_info_cached, **kwargs)
    )
    infos = dict(infos)
    num_fail = 0
    for i in image_ids:
        if infos[i] is None:
            del infos[i]
            num_fail += 1
    return infos, num_fail


async def fetch_images_pixels(image_urls, downloader, dir_, overwrite=False):
    tasks = []
    for i, url in image_urls:
        path = dir_ / image_filename.format(image_id=i)
        if overwrite:
            path.unlink(missing_ok=True)
        tasks.append(downloader.download_image_pixels_cached(url, path))
    num_fail = 0
    for task in tqdm.asyncio.tqdm.as_completed(tasks):
        success = await task
        num_fail += not success
    return num_fail


def split_bbox(bbox: BoundaryBox) -> tuple[BoundaryBox]:
    midpoint = bbox.center
    return (
        BoundaryBox(bbox.min_, midpoint),
        BoundaryBox((bbox.min_[0], midpoint[1]), (midpoint[0], bbox.max_[1])),
        BoundaryBox((midpoint[0], bbox.min_[1]), (bbox.max_[0], midpoint[1])),
        BoundaryBox(midpoint, bbox.max_),
    )


async def fetch_image_list(
    query_bbox: BoundaryBox,
    downloader: MapillaryDownloader,
    **filters,
) -> Tuple[List[int], List[BoundaryBox]]:
    """Because of the limit in number of returned results, we recursively split
    the query area until each query is below this limit.
    """
    pool = [query_bbox]
    finished = []
    all_ids = []
    while len(pool):
        rets = await fetch_many(pool, partial(downloader.get_image_list, **filters))
        pool = []
        for bbox, ids in rets:
            assert ids is not None
            if len(ids) == downloader.max_num_results:
                pool.extend(split_bbox(bbox))
            else:
                finished.append(bbox)
                all_ids.extend(ids)
    return all_ids, finished


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
