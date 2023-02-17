import collections
import collections.abc
import os.path as osp
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pytorch_lightning as pl
import stl.lightning.io.filesystem as stlfs
import torch
import torch.utils.data as torchdata
from omegaconf import OmegaConf  # @manual
from scipy.spatial.transform import Rotation

from .. import logger
from ..osm.tiling_v2 import TileManager
from ..osm.viz import plot_locations
from ..utils.geo import BoundaryBox, Projection
from .dataset import MapLocDataset
from .sequential import chunk_sequence_v2
from .torch import collate, worker_init_fn


def parse_gps_file(path, projection: Projection = None):
    with open(path, "r") as fid:
        lat, lon, _, roll, pitch, yaw, *_ = map(float, fid.read().split())
    latlon = np.array([lat, lon])
    R_world_gps = Rotation.from_euler("ZYX", [yaw, pitch, roll]).as_matrix()
    t_world_gps = None if projection is None else np.r_[projection.project(latlon), 0]
    return latlon, R_world_gps, t_world_gps


def parse_calibration_file(path):
    calib = {}
    with open(path, "r") as fid:
        for line in fid.read().split("\n"):
            if not line:
                continue
            key, *data = line.split(" ")
            key = key.rstrip(":")
            if key.startswith("R"):
                data = np.array(data, float).reshape(3, 3)
            elif key.startswith("T"):
                data = np.array(data, float).reshape(3)
            elif key.startswith("P"):
                data = np.array(data, float).reshape(3, 4)
            calib[key] = data
    return calib


def get_camera_calibration(calib_dir, cam_index: int):
    calib_path = calib_dir / "calib_cam_to_cam.txt"
    calib_cam = parse_calibration_file(calib_path)
    P = calib_cam[f"P_rect_{cam_index:02}"]
    K = P[:3, :3]
    size = np.array(calib_cam[f"S_rect_{cam_index:02}"], float).astype(int)
    camera = {
        "model": "PINHOLE",
        "width": size[0],
        "height": size[1],
        "params": K[[0, 1, 0, 1], [0, 1, 2, 2]],
    }

    t_cam_cam0 = P[:3, 3] / K[[0, 1, 2], [0, 1, 2]]
    R_rect_cam0 = calib_cam["R_rect_00"]

    calib_gps_velo = parse_calibration_file(calib_dir / "calib_imu_to_velo.txt")
    calib_velo_cam0 = parse_calibration_file(calib_dir / "calib_velo_to_cam.txt")
    R_cam0_gps = calib_velo_cam0["R"] @ calib_gps_velo["R"]
    t_cam0_gps = calib_velo_cam0["R"] @ calib_gps_velo["T"] + calib_velo_cam0["T"]
    R_cam_gps = R_rect_cam0 @ R_cam0_gps
    t_cam_gps = t_cam_cam0 + R_rect_cam0 @ t_cam0_gps
    return camera, R_cam_gps, t_cam_gps


class KittiDataModule(pl.LightningDataModule):
    default_cfg = {
        **MapLocDataset.default_cfg,
        "name": "kitti",
        # paths and fetch
        "remote_archive": "manifold://psarlin/tree/maploc/data/kitti_train.tar.gz",
        "local_dir": "/tmp/maploc/kitti",
        "tiles_filename": "tiles.pkl",
        "splits": {
            "train": "train_files.txt",
            "val": "test1_files.txt",
            "test": "test2_files.txt",
        },
        "loading": {
            "train": "???",
            "val": "${.test}",
            "test": {"batch_size": 1, "num_workers": 0},
        },
        "max_num_val": 500,
        "selection_subset_val": "furthest",
        "drop_train_too_close_to_val": 5.0,
        "skip_frames": 1,
        "camera_index": 2,
        # overwrite
        "crop_size_meters": 64,
        "max_init_error": 20,
        "max_init_error_rotation": 10,
        "add_map_mask": True,
        "mask_pad": 2,
        "target_focal_length": 256,
    }
    dummy_scene_name = "kitti"

    def __init__(self, cfg, tile_manager: Optional[TileManager] = None):
        super().__init__()
        default_cfg = OmegaConf.create(self.default_cfg)
        OmegaConf.set_struct(default_cfg, True)  # cannot add new keys
        self.cfg = OmegaConf.merge(default_cfg, cfg)
        self.root = Path(self.cfg.local_dir)
        self.tile_manager = tile_manager
        if self.cfg.crop_size_meters < self.cfg.max_init_error:
            raise ValueError("The ground truth location can be outside the map.")
        assert self.cfg.selection_subset_val in ["random", "furthest"]
        self.splits = {}
        self.shifts = {}
        self.calibrations = {}
        self.data = {}
        self.image_paths = {}

    def prepare_data(self):
        fs = stlfs.get_filesystem(self.cfg.remote_archive)
        assert fs.exists(self.cfg.remote_archive), self.cfg.remote_archive
        if not self.root.exists():
            self.root.parent.mkdir(exist_ok=True, parents=True)
            local_archive = self.root.parent / osp.basename(self.cfg.remote_archive)
            if not local_archive.exists():
                logger.info("Downloading the data from %s", self.cfg.remote_archive)
                fs.get(self.cfg.remote_archive, str(local_archive))
            logger.info("Extracting the data.")
            with tarfile.open(local_archive) as fp:
                fp.extractall(self.root.parent)

    def parse_split(self, split_arg):
        if isinstance(split_arg, str):
            with open(self.root / split_arg, "r") as fid:
                info = fid.read()
            names = []
            shifts = []
            for line in info.split("\n"):
                if not line:
                    continue
                name, *shift = line.split()
                names.append(tuple(name.split("/")))
                if len(shift) > 0:
                    assert len(shift) == 3
                    shifts.append(np.array(shift, float))
            shifts = None if len(shifts) == 0 else np.stack(shifts)
        elif isinstance(split_arg, collections.abc.Sequence):
            names = []
            shifts = None
            for date_drive in split_arg:
                data_dir = (
                    self.root / date_drive / f"image_{self.cfg.camera_index:02}/data"
                )
                assert data_dir.exists(), data_dir
                date_drive = tuple(date_drive.split("/"))
                n = sorted(date_drive + (p.name,) for p in data_dir.glob("*.png"))
                names.extend(n[:: self.cfg.skip_frames])
        else:
            raise ValueError(split_arg)
        return names, shifts

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            stages = ["train", "val"]
        elif stage is None:
            stages = ["train", "val", "test"]
        else:
            stages = [stage]
        for stage in stages:
            self.splits[stage], self.shifts[stage] = self.parse_split(
                self.cfg.splits[stage]
            )
        do_val_subset = "val" in stages and self.cfg.max_num_val is not None
        if do_val_subset and self.cfg.selection_subset_val == "random":
            select = np.random.RandomState(self.cfg.seed).choice(
                len(self.splits["val"]), self.cfg.max_num_val, replace=False
            )
            self.splits["val"] = [self.splits["val"][i] for i in select]
            if self.shifts["val"] is not None:
                self.shifts["val"] = self.shifts["val"][select]
        dates = {d for ns in self.splits.values() for d, _, _ in ns}
        for d in dates:
            self.calibrations[d] = get_camera_calibration(
                self.root / d, self.cfg.camera_index
            )
        if self.tile_manager is None:
            logger.info("Loading the tile manager...")
            self.tile_manager = TileManager.load(self.root / self.cfg.tiles_filename)
        self.cfg.num_classes = {k: len(g) for k, g in self.tile_manager.groups.items()}
        self.cfg.pixel_per_meter = self.tile_manager.ppm

        # pack all attributes in a single tensor to optimize memory access
        self.pack_data(stages)

        dists = None
        if do_val_subset and self.cfg.selection_subset_val == "furthest":
            dists = torch.cdist(
                self.data["val"]["t_c2w"][:, :2].double(),
                self.data["train"]["t_c2w"][:, :2].double(),
            )
            min_dists = dists.min(1).values
            select = torch.argsort(min_dists)[-self.cfg.max_num_val :]
            dists = dists[select]
            self.splits["val"] = [self.splits["val"][i] for i in select]
            if self.shifts["val"] is not None:
                self.shifts["val"] = self.shifts["val"][select]
            for k in list(self.data["val"]):
                if k != "cameras":
                    self.data["val"][k] = self.data["val"][k][select]
            self.image_paths["val"] = self.image_paths["val"][select]

        if "train" in stages and self.cfg.drop_train_too_close_to_val is not None:
            if dists is None:
                dists = torch.cdist(
                    self.data["val"]["t_c2w"][:, :2].double(),
                    self.data["train"]["t_c2w"][:, :2].double(),
                )
            drop = torch.any(dists < self.cfg.drop_train_too_close_to_val, 0)
            select = torch.where(~drop)[0]
            logger.info(
                "Dropping %d (%f %%) images that are too close to validation images.",
                drop.sum(),
                drop.float().mean(),
            )
            self.splits["train"] = [self.splits["train"][i] for i in select]
            if self.shifts["train"] is not None:
                self.shifts["train"] = self.shifts["train"][select]
            for k in list(self.data["train"]):
                if k != "cameras":
                    self.data["train"][k] = self.data["train"][k][select]
            self.image_paths["train"] = self.image_paths["train"][select]

    def pack_data(self, stages):
        for stage in stages:
            names = []
            data = {}
            for i, (date, drive, index) in enumerate(self.splits[stage]):
                d = self.get_frame_data(date, drive, index)
                for k, v in d.items():
                    if i == 0:
                        data[k] = []
                    data[k].append(v)
                path = f"{date}/{drive}/image_{self.cfg.camera_index:02}/data/{index}"
                names.append((self.dummy_scene_name, f"{date}/{drive}", path))
            for k in list(data):
                data[k] = torch.from_numpy(np.stack(data[k]))
            data["camera_id"] = np.full(len(names), self.cfg.camera_index)

            sequences = {date_drive for _, date_drive, _ in names}
            data["cameras"] = {
                self.dummy_scene_name: {
                    seq: {
                        self.cfg.camera_index: self.calibrations[seq.split("/")[0]][0]
                    }
                    for seq in sequences
                }
            }
            if self.shifts[stage] is not None:
                data["shifts"] = torch.from_numpy(self.shifts[stage].astype(np.float32))
            self.data[stage] = data
            self.image_paths[stage] = np.array(names)

    def get_frame_data(self, date, drive, index):
        _, R_cam_gps, t_cam_gps = self.calibrations[date]

        # Transform the GPS pose to the camera pose
        gps_path = (
            self.root / date / drive / "oxts/data" / Path(index).with_suffix(".txt")
        )
        _, R_world_gps, t_world_gps = parse_gps_file(
            gps_path, self.tile_manager.projection
        )
        R_world_cam = R_world_gps @ R_cam_gps.T
        t_world_cam = t_world_gps - R_world_gps @ R_cam_gps.T @ t_cam_gps
        # Some voodoo to extract correct Euler angles from R_world_cam
        R_cv_xyz = Rotation.from_euler("YX", [-90, 90], degrees=True).as_matrix()
        R_world_cam_xyz = R_world_cam @ R_cv_xyz
        y, p, r = Rotation.from_matrix(R_world_cam_xyz).as_euler("ZYX", degrees=True)
        roll, pitch, yaw = r, -p, 90 - y
        roll_pitch_yaw = np.array([-roll, -pitch, yaw], np.float32)  # for some reason

        return {
            "t_c2w": t_world_cam.astype(np.float32),
            "roll_pitch_yaw": roll_pitch_yaw,
        }

    def dataset(self, stage: str):
        return MapLocDataset(
            stage,
            self.cfg,
            self.image_paths[stage],
            self.data[stage],
            {self.dummy_scene_name: self.root},
            {self.dummy_scene_name: self.tile_manager},
        )

    def dataloader(self, stage: str, shuffle: bool = False, num_workers: int = None):
        dataset = self.dataset(stage)
        cfg = self.cfg["loading"][stage]
        num_workers = cfg["num_workers"] if num_workers is None else num_workers
        loader = torchdata.DataLoader(
            dataset,
            batch_size=cfg["batch_size"],
            num_workers=num_workers,
            shuffle=shuffle or (stage == "train"),
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
        )
        return loader

    def train_dataloader(self, **kwargs):
        return self.dataloader("train", **kwargs)

    def val_dataloader(self, **kwargs):
        return self.dataloader("val", **kwargs)

    def test_dataloader(self, **kwargs):
        return self.dataloader("test", **kwargs)

    def sequence_dataset(self, stage: str, chunk_args: Dict = None):
        chunk_args = chunk_args or {}
        keys = self.image_paths[stage]
        key2idx = dict(zip(map(tuple, keys), range(len(keys))))
        # group images by sequence (date/drive)
        seq2names = defaultdict(list)
        for s, date_drive, name in keys:
            seq2names[date_drive].append((name, key2idx[s, date_drive, name]))
        seq2indices = {k: [i for _, i in ns] for k, ns in seq2names.items()}
        # chunk the sequences to the required length
        chunk2indices = {}
        for key, indices in seq2indices.items():
            chunks = chunk_sequence_v2(
                self.data[stage], self.image_paths[stage], indices, **chunk_args
            )
            for i, sub_indices in enumerate(chunks):
                chunk2indices[key, i] = sub_indices
        # store the index of each chunk in its sequence
        chunk_indices = torch.full((len(keys),), -1)
        for (_, chunk_index), idx in chunk2indices.items():
            chunk_indices[idx] = chunk_index
        self.data[stage]["chunk_index"] = chunk_indices
        dataset = self.dataset(stage)
        return dataset, chunk2indices

    def sequence_dataloader(self, stage: str, shuffle: bool = False, **kwargs):
        dataset, chunk2idx = self.sequence_dataset(stage, **kwargs)
        seq_keys = sorted(chunk2idx)
        if shuffle:
            perm = torch.randperm(len(seq_keys))
            seq_keys = [seq_keys[i] for i in perm]
        key_indices = [i for key in seq_keys for i in chunk2idx[key]]
        num_workers = self.cfg["loading"][stage]["num_workers"]
        loader = torchdata.DataLoader(
            dataset,
            batch_size=None,
            sampler=key_indices,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
            collate_fn=collate,
        )
        return loader, seq_keys, chunk2idx


def prepare_osm(
    root,
    sequences,
    output_path=None,
    tile_margin=1000,
    tile_size=128,
    ppm=2,
    osm_path="data/osm/karlsruhe.osm",
):
    all_latlon = []
    for seq in sequences:
        date = "_".join(seq.split("_")[:3])
        gps_paths = (root / date / (seq + "_sync") / "oxts/data").glob("*.txt")
        for p in gps_paths:
            all_latlon.append(parse_gps_file(p)[0])
    all_latlon = np.array(all_latlon)
    print(all_latlon.min(0).tolist(), all_latlon.max(0).tolist())
    # 8.237957,48.902699,8.576459,49.107789

    projection = Projection.mercator(all_latlon)
    all_xy = projection.project(all_latlon)
    bbox_map = BoundaryBox(all_xy.min(0), all_xy.max(0)) + tile_margin
    plot_locations(all_xy, bbox_map)

    if output_path is None:
        output_path = root / KittiDataModule.default_cfg["tiles_filename"]
    osm_path = Path(osm_path)
    if not osm_path.exists():
        raise FileNotFoundError(f"No .osm file at {osm_path}")
    tile_manager = TileManager.from_bbox(
        osm_path,
        projection,
        bbox_map,
        tile_size,
        ppm,
    )
    tile_manager.save(output_path)
    return tile_manager
