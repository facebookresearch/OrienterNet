# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
from collections import defaultdict
import os
import shutil
import tarfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data as torchdata
from omegaconf import DictConfig, OmegaConf

from ... import logger, DATASETS_PATH
from ...osm.tiling import TileManager
from ..dataset import MapLocDataset
from ..sequential import chunk_sequence
from ..torch import collate, worker_init_fn


def pack_dump_dict(dump):
    for per_seq in dump.values():
        if "points" in per_seq:
            for chunk in list(per_seq["points"]):
                points = per_seq["points"].pop(chunk)
                if points is not None:
                    per_seq["points"][chunk] = np.array(
                        per_seq["points"][chunk], np.float64
                    )
        for view in per_seq["views"].values():
            for k in ["R_c2w", "roll_pitch_yaw"]:
                view[k] = np.array(view[k], np.float32)
            for k in ["chunk_id"]:
                if k in view:
                    view.pop(k)
        if "observations" in view:
            view["observations"] = np.array(view["observations"])
        for camera in per_seq["cameras"].values():
            for k in ["params"]:
                camera[k] = np.array(camera[k], np.float32)
    return dump


class MapillaryDataModule(pl.LightningDataModule):
    dump_filename = "dump.json"
    images_archive = "images.tar.gz"
    images_dirname = "images/"

    default_cfg = {
        **MapLocDataset.default_cfg,
        "name": "mapillary",
        # paths and fetch
        "data_dir": DATASETS_PATH / "MGL",
        "local_dir": None,
        "tiles_filename": "tiles.pkl",
        "scenes": "???",
        "split": None,
        "loading": {
            "train": "???",
            "val": "${.test}",
            "test": {"batch_size": 1, "num_workers": 0},
        },
        "filter_for": None,
        "filter_by_ground_angle": None,
        "min_num_points": "???",
    }

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        default_cfg = OmegaConf.create(self.default_cfg)
        OmegaConf.set_struct(default_cfg, True)  # cannot add new keys
        self.cfg = OmegaConf.merge(default_cfg, cfg)
        self.root = Path(self.cfg.data_dir)
        self.local_dir = self.cfg.local_dir or os.environ.get("TMPDIR")
        if self.local_dir is not None:
            self.local_dir = Path(self.local_dir, "MGL")
        if self.cfg.crop_size_meters < self.cfg.max_init_error:
            raise ValueError("The ground truth location can be outside the map.")

    def prepare_data(self):
        for scene in self.cfg.scenes:
            dump_dir = self.root / scene
            assert (dump_dir / self.dump_filename).exists(), dump_dir
            assert (dump_dir / self.cfg.tiles_filename).exists(), dump_dir
            if self.local_dir is None:
                assert (dump_dir / self.images_dirname).exists(), dump_dir
                continue
            # Cache the folder of images locally to speed up reading
            local_dir = self.local_dir / scene
            if local_dir.exists():
                shutil.rmtree(local_dir)
            local_dir.mkdir(exist_ok=True, parents=True)
            images_archive = dump_dir / self.images_archive
            logger.info("Extracting the image archive %s.", images_archive)
            with tarfile.open(images_archive) as fp:
                fp.extractall(local_dir)

    def setup(self, stage: Optional[str] = None):
        self.dumps = {}
        self.tile_managers = {}
        self.image_dirs = {}
        names = []

        for scene in self.cfg.scenes:
            logger.info("Loading scene %s.", scene)
            dump_dir = self.root / scene
            logger.info("Loading map tiles %s.", self.cfg.tiles_filename)
            self.tile_managers[scene] = TileManager.load(
                dump_dir / self.cfg.tiles_filename
            )
            groups = self.tile_managers[scene].groups
            if self.cfg.num_classes:  # check consistency
                if set(groups.keys()) != set(self.cfg.num_classes.keys()):
                    raise ValueError(
                        f"Inconsistent groups: {groups.keys()} {self.cfg.num_classes.keys()}"
                    )
                for k in groups:
                    if len(groups[k]) != self.cfg.num_classes[k]:
                        raise ValueError(
                            f"{k}: {len(groups[k])} vs {self.cfg.num_classes[k]}"
                        )
            ppm = self.tile_managers[scene].ppm
            if ppm != self.cfg.pixel_per_meter:
                raise ValueError(
                    "The tile manager and the config/model have different ground resolutions: "
                    f"{ppm} vs {self.cfg.pixel_per_meter}"
                )

            logger.info("Loading dump json file %s.", self.dump_filename)
            with (dump_dir / self.dump_filename).open("r") as fp:
                self.dumps[scene] = pack_dump_dict(json.load(fp))
            for seq, per_seq in self.dumps[scene].items():
                for cam_id, cam_dict in per_seq["cameras"].items():
                    if cam_dict["model"] != "PINHOLE":
                        raise ValueError(
                            f"Unsupported camera model: {cam_dict['model']} for {scene},{seq},{cam_id}"
                        )

            self.image_dirs[scene] = (
                (self.local_dir or self.root) / scene / self.images_dirname
            )
            assert self.image_dirs[scene].exists(), self.image_dirs[scene]

            for seq, data in self.dumps[scene].items():
                for name in data["views"]:
                    names.append((scene, seq, name))

        self.parse_splits(self.cfg.split, names)
        if self.cfg.filter_for is not None:
            self.filter_elements()
        self.pack_data()

    def pack_data(self):
        # We pack the data into compact tensors that can be shared across processes without copy
        exclude = {
            "compass_angle",
            "compass_accuracy",
            "gps_accuracy",
            "chunk_key",
            "panorama_offset",
        }
        cameras = {
            scene: {seq: per_seq["cameras"] for seq, per_seq in per_scene.items()}
            for scene, per_scene in self.dumps.items()
        }
        points = {
            scene: {
                seq: {
                    i: torch.from_numpy(p) for i, p in per_seq.get("points", {}).items()
                }
                for seq, per_seq in per_scene.items()
            }
            for scene, per_scene in self.dumps.items()
        }
        self.data = {}
        for stage, names in self.splits.items():
            view = self.dumps[names[0][0]][names[0][1]]["views"][names[0][2]]
            data = {k: [] for k in view.keys() - exclude}
            for scene, seq, name in names:
                for k in data:
                    data[k].append(self.dumps[scene][seq]["views"][name].get(k, None))
            for k in data:
                v = np.array(data[k])
                if np.issubdtype(v.dtype, np.integer) or np.issubdtype(
                    v.dtype, np.floating
                ):
                    v = torch.from_numpy(v)
                data[k] = v
            data["cameras"] = cameras
            data["points"] = points
            self.data[stage] = data
            self.splits[stage] = np.array(names)

    def filter_elements(self):
        for stage, names in self.splits.items():
            names_select = []
            for scene, seq, name in names:
                view = self.dumps[scene][seq]["views"][name]
                if self.cfg.filter_for == "ground_plane":
                    if not (1.0 <= view["height"] <= 3.0):
                        continue
                    planes = self.dumps[scene][seq].get("plane")
                    if planes is not None:
                        inliers = planes[str(view["chunk_id"])][-1]
                        if inliers < 10:
                            continue
                    if self.cfg.filter_by_ground_angle is not None:
                        plane = np.array(view["plane_params"])
                        normal = plane[:3] / np.linalg.norm(plane[:3])
                        angle = np.rad2deg(np.arccos(np.abs(normal[-1])))
                        if angle > self.cfg.filter_by_ground_angle:
                            continue
                elif self.cfg.filter_for == "pointcloud":
                    if len(view["observations"]) < self.cfg.min_num_points:
                        continue
                elif self.cfg.filter_for is not None:
                    raise ValueError(f"Unknown filtering: {self.cfg.filter_for}")
                names_select.append((scene, seq, name))
            logger.info(
                "%s: Keep %d/%d images after filtering for %s.",
                stage,
                len(names_select),
                len(names),
                self.cfg.filter_for,
            )
            self.splits[stage] = names_select

    def parse_splits(self, split_arg, names):
        if split_arg is None:
            self.splits = {
                "train": names,
                "val": names,
            }
        elif isinstance(split_arg, int):
            names = np.random.RandomState(self.cfg.seed).permutation(names).tolist()
            self.splits = {
                "train": names[split_arg:],
                "val": names[:split_arg],
            }
        elif isinstance(split_arg, DictConfig):
            scenes_val = set(split_arg.val)
            scenes_train = set(split_arg.train)
            assert len(scenes_val - set(self.cfg.scenes)) == 0
            assert len(scenes_train - set(self.cfg.scenes)) == 0
            self.splits = {
                "train": [n for n in names if n[0] in scenes_train],
                "val": [n for n in names if n[0] in scenes_val],
            }
        elif isinstance(split_arg, str):
            with (self.root / split_arg).open("r") as fp:
                splits = json.load(fp)
            splits = {
                k: {loc: set(ids) for loc, ids in split.items()}
                for k, split in splits.items()
            }
            self.splits = {}
            for k, split in splits.items():
                self.splits[k] = [
                    n
                    for n in names
                    if n[0] in split and int(n[-1].rsplit("_", 1)[0]) in split[n[0]]
                ]
        else:
            raise ValueError(split_arg)

    def dataset(self, stage: str):
        return MapLocDataset(
            stage,
            self.cfg,
            self.splits[stage],
            self.data[stage],
            self.image_dirs,
            self.tile_managers,
            image_ext=".jpg",
        )

    def dataloader(
        self,
        stage: str,
        shuffle: bool = False,
        num_workers: int = None,
        sampler: Optional[torchdata.Sampler] = None,
    ):
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
            sampler=sampler,
        )
        return loader

    def train_dataloader(self, **kwargs):
        return self.dataloader("train", **kwargs)

    def val_dataloader(self, **kwargs):
        return self.dataloader("val", **kwargs)

    def test_dataloader(self, **kwargs):
        return self.dataloader("test", **kwargs)

    def sequence_dataset(self, stage: str, **kwargs):
        keys = self.splits[stage]
        seq2indices = defaultdict(list)
        for index, (_, seq, _) in enumerate(keys):
            seq2indices[seq].append(index)
        # chunk the sequences to the required length
        chunk2indices = {}
        for seq, indices in seq2indices.items():
            chunks = chunk_sequence(self.data[stage], indices, **kwargs)
            for i, sub_indices in enumerate(chunks):
                chunk2indices[seq, i] = sub_indices
        # store the index of each chunk in its sequence
        chunk_indices = torch.full((len(keys),), -1)
        for (_, chunk_index), idx in chunk2indices.items():
            chunk_indices[idx] = chunk_index
        self.data[stage]["chunk_index"] = chunk_indices
        dataset = self.dataset(stage)
        return dataset, chunk2indices

    def sequence_dataloader(self, stage: str, shuffle: bool = False, **kwargs):
        dataset, chunk2idx = self.sequence_dataset(stage, **kwargs)
        chunk_keys = sorted(chunk2idx)
        if shuffle:
            perm = torch.randperm(len(chunk_keys))
            chunk_keys = [chunk_keys[i] for i in perm]
        key_indices = [i for key in chunk_keys for i in chunk2idx[key]]
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
        return loader, chunk_keys, chunk2idx
