# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import json
import os.path as osp
import pickle
import tarfile
from pathlib import Path
from typing import List, Optional

import fsspec
import numpy as np
import pytorch_lightning as pl
import stl.lightning.io.filesystem as stlfs
import torch
import torch.utils.data as torchdata
import torchvision.transforms as tvf
from omegaconf import OmegaConf  # @manual
from PIL import Image

from ... import logger
from ...utils.io import read_image
from ...utils.wrappers import Camera
from ..collate import collate
from ..image import rectify_image, resize_image
from ..utils import crop_map, decompose_rotmat, random_flip, random_rot90


class MapLocDataset(torchdata.Dataset):
    def __init__(self, stage: str, cfg, names: List[str], dump, images, rasters):
        self.stage = stage
        self.cfg = cfg
        self.names = names
        self.dump = dump
        self.images = images
        self.rasters = rasters
        self.ppm = dump["raster"]["ppm"]
        if self.ppm != cfg.pixel_per_meter:
            raise ValueError(
                "The data dump and the config/model have different ground resolutions: "
                f"{self.ppm} vs {cfg.pixel_per_meter}"
            )

        tfs = [tvf.ToTensor()]
        if stage == "train" and cfg.augmentation.image.apply:
            args = OmegaConf.masked_copy(
                cfg.augmentation.image, ["brightness", "contrast", "saturation", "hue"]
            )
            tfs.append(tvf.ColorJitter(**args))
        self.tfs = tvf.Compose(tfs)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        view = self.dump["views"][name]
        if self.stage == "train" and self.cfg.random:
            seed = None
        else:
            seed = self.cfg.seed + idx

        cam_id = view["camera_id"]
        cam_dict = self.dump["cameras"][str(cam_id)]
        assert cam_dict["model"] == "PINHOLE"
        cam = Camera.from_colmap(cam_dict).float()

        roll, pitch, yaw = decompose_rotmat(view["R_c2w"])
        height = view.get("height") or self.cfg.camera_height

        image = read_image(self.images / name)
        raster = np.asarray(Image.open(self.rasters[name])).copy().transpose(2, 0, 1)
        xy = np.array(raster.shape[:2][::-1]) / 2 - 0.5  # center of the raster

        # Map crop
        crop_size = self.ppm * self.cfg.crop_size_meters * 2
        raster, xy = crop_map(raster, xy, crop_size, seed=seed)

        # Map augmentations
        # TODO: handle K
        heading = np.deg2rad(90 - yaw)  # fixme
        if self.stage == "train":
            if self.cfg.augmentation.rot90:
                raster, xy, heading = random_rot90(raster, xy, heading)
            if self.cfg.augmentation.flip:
                image, raster, xy, heading = random_flip(image, raster, xy, heading)
        yaw = 90 - np.rad2deg(heading)  # fixme

        if self.cfg.resize_image is not None:
            image, _, cam = resize_image(image, self.cfg.resize_image, camera=cam)

        # Convert to torch
        image = self.tfs(np.ascontiguousarray(image))
        raster = torch.from_numpy(np.ascontiguousarray(raster)).long()

        image = rectify_image(
            image, cam, roll, pitch if self.cfg.rectify_pitch else None
        )
        roll = 0.0
        if self.cfg.rectify_pitch:
            pitch = 0.0

        return {
            "name": name,
            "image": image,
            "map": raster,
            "xy": torch.from_numpy(xy).float(),
            "roll_pitch_yaw": torch.tensor((roll, pitch, yaw)).float(),
            "camera_height": torch.tensor(height).float(),
            "pixels_per_meter": torch.tensor(self.ppm).float(),
            "camera": cam,
        }


class MapLocDataModule(pl.LightningDataModule):
    dump_filename = "dump.json"
    rasters_filename = "rasters.pkl"
    images_archive = "images.tar.gz"
    images_dirname = "images/"

    default_cfg = {
        "seed": 0,
        "random": True,
        "split": None,
        "dump_dir": "???",
        "local_dir": "/tmp/maploc/",
        "num_classes": "???",
        "train": "???",
        "val": "???",
        "crop_size_meters": "???",
        "resize_image": None,
        "rectify_pitch": False,
        "camera_height": "???",
        "pixel_per_meter": "???",
        "augmentation": {
            "rot90": False,
            "flip": False,
            "image": {
                "apply": False,
                "brightness": 0.5,
                "contrast": 0.4,
                "saturation": 0.4,
                "hue": 0.5 / 3.14,
            },
        },
    }

    def __init__(self, cfg):
        super().__init__()
        default_cfg = OmegaConf.create(self.default_cfg)
        OmegaConf.set_struct(default_cfg, True)  # cannot add new keys
        self.cfg = OmegaConf.merge(default_cfg, cfg)
        self.root = self.cfg.dump_dir
        self.local_dir = Path(self.cfg.local_dir)

    def prepare_data(self):
        fs = stlfs.get_filesystem(self.root)
        assert fs.exists(osp.join(self.root, self.dump_filename)), self.root
        assert fs.exists(osp.join(self.root, self.rasters_filename)), self.root

        # Cache the folder of images locally to speed up reading
        images = self.local_dir / self.images_dirname
        if not images.exists():
            images_archive = self.local_dir / self.images_archive
            if not images_archive.exists():
                logger.info("Downloading the image archive from manifold.")
                images_archive.parent.mkdir(exist_ok=True, parents=True)
                images_archive_remote = osp.join(self.root, self.images_archive)
                fs = stlfs.get_filesystem(images_archive_remote)
                fs.get(images_archive_remote, str(images_archive))
            logger.info("Extracting the image archive.")
            with tarfile.open(images_archive) as fp:
                fp.extractall(self.local_dir)

    def setup(self, stage: Optional[str] = None):
        logger.info("Loading dump json file %s.", self.dump_filename)
        with fsspec.open(osp.join(self.root, self.dump_filename), "r") as fp:
            self.dump = json.load(fp)
        groups = self.dump["raster"]["groups"]

        # Check the number of classes
        if set(groups.keys()) != set(self.cfg.num_classes.keys()):
            raise ValueError(
                f"Inconsistent groups: {groups.keys()} {self.cfg.num_classes.keys()}"
            )
        for k in groups:
            if len(groups[k]) != self.cfg.num_classes[k]:
                raise ValueError(f"{k}: {len(groups[k])} vs {self.cfg.num_classes[k]}")
        if self.dump["raster"]["tile_radius"] > 2 * self.cfg.crop_size_meters:
            raise ValueError("No margin between the tile center and the crop bbox.")

        logger.info("Loading rasters pickle %s.", self.rasters_filename)
        with fsspec.open(osp.join(self.root, self.rasters_filename), "rb") as fp:
            self.rasters = pickle.load(fp)

        # Read from the local cache or the source directory
        self.images = self.local_dir / self.images_dirname
        assert self.images.exists(), self.images

        names = sorted(self.dump["views"])
        if self.cfg.split is None:
            self.splits = {
                "train": names,
                "val": names,
            }
        elif isinstance(self.cfg.split, int):
            names = np.random.RandomState(self.cfg.seed).permutation(names).tolist()
            self.splits = {
                "train": names[self.cfg.split :],
                "val": names[: self.cfg.split],
            }
        elif isinstance(self.cfg.split, str):
            with fsspec.open(osp.join(self.root, self.cfg.split), "r") as fp:
                self.splits = json.load(fp)
        else:
            raise ValueError(self.cfg.split)

    def dataloader(self, stage: str, shuffle: bool = False):
        dataset = MapLocDataset(
            stage, self.cfg, self.splits[stage], self.dump, self.images, self.rasters
        )
        loader = torchdata.DataLoader(
            dataset,
            batch_size=self.cfg[stage]["batch_size"],
            num_workers=self.cfg[stage]["num_workers"],
            shuffle=shuffle or (stage == "train"),
            pin_memory=True,
            persistent_workers=True,
            collate_fn=collate,
        )
        return loader

    def train_dataloader(self):
        return self.dataloader("train")

    def val_dataloader(self):
        return self.dataloader("val")
