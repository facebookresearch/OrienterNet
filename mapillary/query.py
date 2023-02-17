# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import concurrent.futures
import json
import logging
import os.path as osp
from pathlib import Path
from typing import Dict, List

from mapillary.data_migration.tools.query_mapillary import (
    await_sync,
    download_data,
    MapillaryDownloadParams,
    run_query,
    sql_int_set,
    sql_string_set,
    subset_data_from_chunk_ids,
    SubsetData,
)
from mapillary.vision.common.config.buckets import THUMBS_BUCKET
from mapillary.vision.common.utils import fileio
from tqdm import tqdm

from ..utils.geo import BoundaryBox
from ..utils.io import write_json

logger = logging.getLogger(__name__)
timeout = 300


def bbox_condition(bbox: BoundaryBox):
    minlat, minlon = bbox.min_
    maxlat, maxlon = bbox.max_
    return (
        f"(image.latitude >= {minlat} AND image.latitude <= {maxlat} AND "
        f"image.longitude >= {minlon} AND image.longitude <= {maxlon})"
    )


def date_condition(date: str):
    return f"image.ds = '{date}'"


def camera_model_condition(models):
    return f"image.model in {sql_string_set(models)}"


def camera_type_condition(types):
    return f"image.camera_type in {sql_string_set(types)}"


def owner_condition(owners):
    return f"image.owner_key in {sql_string_set(owners)}"


def filter_chunks_by_quality(chunk_ids):
    query = f"""
    SELECT fbid, reconstruction_quality, initial_reconstruction_id
    FROM mly_chunks_xdb_export AS chunk
    WHERE
        chunk.fbid in {sql_int_set(chunk_ids)}
        AND chunk.reconstruction_quality IN (200, 100)
    GROUP BY fbid, reconstruction_quality, initial_reconstruction_id
    """
    results = run_query(query, timeout)
    quality = {r["fbid"]: r["reconstruction_quality"] for r in results}
    has_tracks = {
        r["fbid"]: r["initial_reconstruction_id"] is not None for r in results
    }
    return quality, has_tracks


def filter_max_num_images(chunk_ids: List[int], max_num: int) -> List[int]:
    query = f"""
    SELECT sfm_chunk_id, COUNT(*)
    FROM (
        SELECT DISTINCT image_key, sfm_chunk_id
        FROM mly_images_xdb_export_signal_converted AS image
        WHERE
            image.sfm_chunk_id in {sql_int_set(chunk_ids)})
    GROUP BY sfm_chunk_id
    """
    results = run_query(query, timeout)
    counts = {r["sfm_chunk_id"]: r["_col1"] for r in results}
    ids_sorted = sorted(chunk_ids, key=lambda i: counts.get(i, 0), reverse=True)
    ids_selected = []
    counter = 0
    for i in ids_sorted:
        if i in counts:
            counter += counts[i]
            ids_selected.append(i)
            if counter >= max_num:
                break
    return ids_selected


def query_chunks(
    bbox,
    camera_models=None,
    camera_types=None,
    date=None,
    owners=None,
    filter_quality=True,
    max_num_images=None,
):
    template = """
    SELECT
        sfm_chunk_id, sfm_cluster_key, sequence_key, owner_key, model
    FROM mly_images_xdb_export_signal_converted AS image
    WHERE {conditions}
    GROUP BY
        sfm_chunk_id, sfm_cluster_key, sequence_key, owner_key, model
    """
    conditions = []
    if date is not None:
        conditions.append(date_condition(date))
    if camera_models is not None:
        conditions.append(camera_model_condition(camera_models))
    if camera_types is None:
        camera_types = ["perspective", "fisheye", "spherical", "equirectangular"]
    conditions.append(camera_type_condition(camera_types))
    if owners is not None:
        conditions.append(owner_condition(owners))
    conditions.append(bbox_condition(bbox))
    conditions = " AND ".join(conditions)

    print(f"Querying with condition {conditions}")
    query = template.format(conditions=conditions)
    results = run_query(query, timeout)
    chunk_ids = list({r["sfm_chunk_id"] for r in results} - {None})
    chunk2data = {
        r["sfm_chunk_id"]: r for r in results if r["sfm_chunk_id"] is not None
    }

    if filter_quality:
        chunk2qual, chunk_has_tracks = filter_chunks_by_quality(chunk_ids)
        chunk_ids = sorted(chunk2qual)
    else:
        chunk_has_tracks = None
    if max_num_images is not None:
        chunk_ids = filter_max_num_images(chunk_ids, max_num_images)

    return chunk_ids, chunk2data, chunk_has_tracks, query


def query_chunks_cached(dump_path: Path, bbox, **kwargs):
    path = dump_path / "query.json"
    if path.exists():
        with path.open() as fp:
            dump = json.load(fp)
        bbox = BoundaryBox.from_string(dump["arguments"]["bbox"])
        chunk_ids, chunk2data = dump["chunk_ids"], dump["chunk2data"]
        chunk2data = {int(k): v for k, v in chunk2data.items() if k != "null"}
        chunk_has_tracks = dump["chunk_has_tracks"]
    else:
        chunk_ids, chunk2data, chunk_has_tracks, _ = query_chunks(bbox, **kwargs)
        dump = {
            "arguments": {
                "bbox": bbox.format(),
                **kwargs,
            },
            "chunk_ids": chunk_ids,
            "chunk2data": chunk2data,
            "chunk_has_tracks": chunk_has_tracks,
        }
        write_json(path, dump)
    return chunk_ids, chunk2data, chunk_has_tracks, dump


def download_thumbnails_parallel(output_dir, sd: SubsetData):
    def download_thumbnail(image_id, image_key):
        thumb_id = sd.image_thumb_id_from_id[image_id]
        src = f"manifold://{THUMBS_BUCKET}/flat/{thumb_id}"
        dst = osp.join(output_dir, "images", f"{image_key}")
        fileio.copy(src, dst)

    fileio.make_dirs(osp.join(output_dir, "images"))
    logger.info("Starting to download thumbnail images...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = [
            executor.submit(download_thumbnail, image_id, image_key)
            for image_id, image_key in sd.image_key_from_id.items()
        ]
        with tqdm(total=len(futures)) as pbar:
            for _ in concurrent.futures.as_completed(futures):
                pbar.update(1)


def thumbnail_key_to_id(sd) -> Dict[str, int]:
    return {k: sd.image_thumb_id_from_id[i] for i, k in sd.image_key_from_id.items()}


def download_chunks(cluster_ids: List[int], out_dir: Path):
    params = MapillaryDownloadParams(
        output_dir=str(out_dir),
        image_vrs=False,
        exif_vrs=True,
        panoptic_vrs=True,
        thumbnails=False,
    )
    sd = await_sync(subset_data_from_chunk_ids(cluster_ids, include_unprocessed=True))
    logger.info("Starting to download cluster data...")
    download_data(sd, params)
    # download_thumbnails_parallel(params.output_dir, sd)
    write_json(out_dir / "thumbnail_key_to_id.json", thumbnail_key_to_id(sd))
