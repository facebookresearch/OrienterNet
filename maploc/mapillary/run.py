# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import argparse
import json
import os.path as osp
import shutil
import tarfile
from copy import deepcopy
from pathlib import Path
from typing import Optional

import stl.lightning.io.filesystem as stlfs
from libfb.py.asyncio.await_utils import await_sync
from manifold.clients.python import ManifoldClient
from omegaconf import DictConfig, OmegaConf  # @manual

from .. import logger
from ..osm.tiling_v2 import TileManager
from ..utils.geo import BoundaryBox, Projection
from ..utils.io import write_json
from .osm_filter import BuildingFilter
from .processing import (
    get_dataset_bbox,
    order_outputs_by_sequence,
    plot_coverage,
    plot_coverage_selection,
    process_all_chunks,
)
from .query import download_chunks, query_chunks_cached
from .reconstruction import ClusterVRSDatasetWithThumbnails


def main(
    dataset_path: Path,
    dump_path: Path,
    bbox: BoundaryBox,
    epsg: str,
    cfg: DictConfig,
    osm_path: Optional[Path] = None,
    **query_kwargs,
):
    dump_path.mkdir(exist_ok=True, parents=True)
    image_dir = dump_path / "images"
    image_dir.mkdir(exist_ok=True)

    (
        chunk_ids,
        chunk2data,
        _,
        _,
    ) = query_chunks_cached(dump_path, bbox, **query_kwargs)
    sequence_keys = list({r["sequence_key"] for r in chunk2data.values()})
    users = {r["owner_key"] for r in chunk2data.values()}
    logger.info(
        f"Valid chunks {len(chunk_ids)}/{len(chunk2data)}, {len(sequence_keys)} sequences, {len(users)} users"
    )

    if not dataset_path.exists():
        logger.info("Downloading the dataset...")
        download_chunks(chunk_ids, dataset_path)
    logger.info("Loading the dataset.")
    dataset = ClusterVRSDatasetWithThumbnails.single_from_path(str(dataset_path))
    with open(dataset_path / "thumbnail_key_to_id.json", "r") as fp:
        dataset.thumbnail_key_to_id = json.load(fp)

    chunk_ids_with_aligned = [
        i
        for i in chunk_ids
        if dataset.cluster_reconstruction_aligned_exists(
            chunk2data[i]["sfm_cluster_key"]
        )
    ]
    logger.info(
        "Only %d/%d clusters with aligned reconstruction.",
        len(chunk_ids_with_aligned),
        len(chunk_ids),
    )
    chunk_ids = chunk_ids_with_aligned

    plot_coverage(dump_path / "coverage_raw.html", dataset, chunk_ids, chunk2data)

    projection = Projection(epsg)
    bbox_total = get_dataset_bbox(dataset, chunk_ids, chunk2data, projection)
    print(bbox_total)
    bbox_tiling = bbox_total + cfg.tiling.margin
    logger.info("Creating the map tiles.")
    tile_manager = TileManager.from_bbox(
        osm_path or dump_path,
        projection,
        bbox_tiling,
        cfg.tiling.tile_size,
        cfg.tiling.ppm,
    )
    tile_manager.save(dump_path / "tiles.pkl")
    building_filter = BuildingFilter(bbox_tiling, tile_manager.map_data, cfg.tiling.ppm)

    logger.info("Starting to process the data...")
    outputs, stats, to_investigate = process_all_chunks(
        dataset, chunk_ids, chunk2data, projection, building_filter, image_dir, cfg
    )
    write_json(dump_path / "outputs_per_cluster.json", outputs)
    write_json(dump_path / "failures.json", to_investigate)

    count = stats.pop("count")
    logger.info({k: v / count for k, v in stats.items()})
    logger.info(sum([len(v["views"]) for v in outputs.values()]))
    outputs_per_sequence = order_outputs_by_sequence(outputs, chunk2data)
    write_json(dump_path / "outputs_per_sequence.json", outputs_per_sequence)

    OmegaConf.save(config=cfg, f=dump_path / "config.yaml")

    plot_coverage_selection(
        dump_path / "coverage_selection.html", dataset, chunk_ids, chunk2data, outputs
    )


def upload_to_manifold(local_root: Path, remote_root: str, scene: str):
    local_dir = local_root / scene
    logger.info("Creating the image archive.")
    with tarfile.open(local_dir / "images.tar.gz", "w:gz") as fp:
        fp.add(local_dir / "images", arcname="images")
    files = {f.name for f in local_dir.iterdir()} - {"images"}
    remote_dir = osp.join(remote_root, scene)
    fs = stlfs.get_filesystem(remote_root)
    fs.makedirs(remote_dir, exist_ok=True)
    bucket, *remote_subdir = remote_dir.replace("manifold://", "").split("/")

    async def put_file(remote_path, local_path):
        with ManifoldClient.get_client(bucket=bucket) as client:
            await client.put(remote_path, str(local_path))

    for filename in files:
        logger.info("Uploading file %s to manifold %s", filename, remote_dir)
        # We need to use Manifold because large file may exceed transfer quotas
        # fsspec does not handle retries and throttling.
        await_sync(put_file(osp.join(*remote_subdir, filename), local_dir / filename))
        # fs.put_file(str(local_dir / filename), osp.join(remote_dir, filename))


date = "2022-06-17"
locations = {
    "sanfrancisco_soma_test": {
        "bbox": BoundaryBox([37.7852, -122.4014], [37.7946, -122.3900]),
        "camera_models": ["GoPro Max"],
        "epsg": "EPSG:7131",
        "osm_file": "sanfrancisco.osm",
    },
    "sanfrancisco_soma": {
        "bbox": BoundaryBox(
            [-122.410307, 37.770364][::-1], [-122.388772, 37.795545][::-1]
        ),
        "camera_models": ["GoPro Max"],
        "epsg": "EPSG:7131",
        "osm_file": "sanfrancisco.osm",
    },
    "sanfrancisco_hayes": {
        "bbox": BoundaryBox(
            [-122.438415, 37.768634][::-1], [-122.410605, 37.783894][::-1]
        ),
        "camera_models": ["GoPro Max"],
        "epsg": "EPSG:7131",
        "osm_file": "sanfrancisco.osm",
    },
    "amsterdam": {
        "bbox": BoundaryBox([4.845284, 52.340679][::-1], [4.926147, 52.386299][::-1]),
        "camera_models": ["GoPro Max"],
        "epsg": "EPSG:28992",
        "osm_file": "amsterdam.osm",
    },
    "lemans": {
        "bbox": BoundaryBox([0.185752, 47.995125][::-1], [0.224088, 48.014209][::-1]),
        "owners": ["xXOocM1jUB4jaaeukKkmgw"],  # sogefi
        "epsg": "EPSG:2154",
        "osm_file": "lemans.osm",
    },
    "berlin": {
        "bbox": BoundaryBox([13.416271, 52.459656][::-1], [13.469829, 52.499195][::-1]),
        "owners": ["LT3ajUxH6qsosamrOHIrFw"],  # supaplex030
        "epsg": "EPSG:5243",
        "osm_file": "berlin.osm",
    },
    "montrouge": {
        "bbox": BoundaryBox([2.298958, 48.80874][::-1], [2.332989, 48.825276][::-1]),
        "owners": [
            "XtzGKZX2_VIJRoiJ8IWRNQ",
            "C4ENdWpJdFNf8CvnQd7NrQ",
            "e_ZBE6mFd7CYNjRSpLl-Lg",
        ],  # overflorian, phyks, francois2
        "camera_models": ["LG-R105"],
        "epsg": "EPSG:2154",
        "osm_file": "paris.osm",
    },
    "nantes": {
        "bbox": BoundaryBox([-1.585839, 47.198289][::-1], [-1.51318, 47.236161][::-1]),
        "owners": [
            "jGdq3CL-9N-Esvj3mtCWew",
            "s-j5BH9JRIzsgORgaJF3aA",
        ],  # c_mobilite, cartocite
        "epsg": "EPSG:2154",
        "osm_file": "nantes.osm",
    },
    "toulouse": {
        "bbox": BoundaryBox([1.429457, 43.591434][::-1], [1.456653, 43.61343][::-1]),
        "owners": ["MNkhq6MCoPsdQNGTMh3qsQ"],  # tyndare
        "epsg": "EPSG:2154",
        "osm_file": "toulouse.osm",
    },
    "vilnius": {
        "bbox": BoundaryBox([25.258633, 54.672956][::-1], [25.296094, 54.696755][::-1]),
        "owners": ["bClduFF6Gq16cfwCdhWivw", "u5ukBseATUS8jUbtE43fcO"],  # kedas, vms
        "epsg": "EPSG:3346",
        "osm_file": "vilnius.osm",
    },
    "newyork_hoboken": {
        "bbox": BoundaryBox(
            [-74.048367, 40.73252][::-1], [-74.017123, 40.761313][::-1]
        ),
        "owners": ["hAQ5Q1gWGLKC33uOjxUWaZ"],  # amidave
        "epsg": "EPSG:2824",
        "osm_file": "hoboken.osm",
        "max_num_images": 60000,
    },
    "helsinki": {
        "bbox": BoundaryBox(
            [24.8975480117, 60.1449128318][::-1], [24.9816543235, 60.1770977471][::-1]
        ),
        "camera_types": ["spherical", "equirectangular"],
        "epsg": "EPSG:3067",
        "osm_file": "helsinki.osm",
    },
    "milan": {
        "bbox": BoundaryBox(
            [9.1732723899, 45.4810977947][::-1],
            [9.2255987917, 45.5284238563][::-1],
        ),
        "camera_types": ["spherical", "equirectangular"],
        "epsg": "EPSG:7794",
        "osm_file": "milan.osm",
    },
    "avignon": {
        "bbox": BoundaryBox(
            [4.7887045302, 43.9416178156][::-1], [4.8227015622, 43.9584848909][::-1]
        ),
        "camera_types": ["spherical", "equirectangular"],
        "epsg": "EPSG:2154",
        "osm_file": "avignon.osm",
    },
    "paris": {
        "bbox": BoundaryBox([2.306823, 48.833827][::-1], [2.39067, 48.889335][::-1]),
        "camera_types": ["spherical", "equirectangular"],
        "epsg": "EPSG:2154",
        "osm_file": "paris.osm",
    },
    "brussels": {
        "bbox": BoundaryBox([4.336402, 50.797682][::-1], [4.390158, 50.859124][::-1]),
        "camera_models": ["GoPro Max"],
        "epsg": "EPSG:31370",
        "osm_file": "brussels.osm",
    },
}
cfg = OmegaConf.create(
    {
        "min_num_images": 5,
        "max_ratio_in_building": 0.3,
        "max_image_size": 512,
        "do_random_pano_offset": True,
        "min_dist_between_keyframes": 4,
        "do_plane_fitting": False,
        "do_retriangulation": False,
        "include_points": False,
        "num_proc": 20,
        "verbose": False,
        "tiling": {
            "tile_size": 128,
            "margin": 128,
            "ppm": 2,
        },
    }
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--locations", type=str, nargs="+", required=True)
    parser.add_argument(
        "--dataset_dir", type=Path, default=Path("/home/psarlin/local/data/mapillary/")
    )
    parser.add_argument(
        "--dump_dir",
        type=Path,
        default=Path("/home/psarlin/local/data/mapillary_dumps_v2/"),
    )
    parser.add_argument(
        "--manifold_dir",
        type=str,
        default="manifold://psarlin/tree/maploc/data/mapillary_v2/",
    )
    parser.add_argument(
        "--osm_dir",
        type=Path,
        default=Path("/data/users/psarlin/data/osm/"),
        help="Can be downloaded from manifold://psarlin/tree/maploc/osm/",
    )
    parser.add_argument("--keep_images", action="store_true")
    args = parser.parse_args()

    for location in args.locations:
        logger.info("Working on location %s", location)
        params = deepcopy(locations[location])
        main(
            args.dataset_dir / location,
            args.dump_dir / location,
            osm_file=args.osm_dir / params.pop("osm"),
            **params,
            date=date,
            cfg=cfg,
        )
        upload_to_manifold(args.dump_dir, args.manifold_dir, location)
        if not args.keep_images:
            image_dir = args.dataset_dir / location / "images"
            if image_dir.exists():
                shutil.rmtree(image_dir)
