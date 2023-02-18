# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.utils.data as torchdata
import torchvision.transforms as tvf
from omegaconf import DictConfig, OmegaConf  # @manual
from scipy.spatial.transform import Rotation

from ..models.utils import deg2rad, rotmat2d
from ..osm.tiling_v2 import TileManager
from ..utils.geo import BoundaryBox
from ..utils.io import read_image
from ..utils.wrappers import Camera
from .image import center_pad_crop_image, pad_image, rectify_image, resize_image
from .utils import decompose_rotmat, random_flip, random_rot90


class MapLocDataset(torchdata.Dataset):
    default_cfg = {
        "seed": 0,
        "random": True,
        "num_threads": None,
        # map
        "num_classes": None,
        "pixel_per_meter": "???",
        "crop_size_meters": "???",
        "max_init_error": "???",
        "max_init_error_rotation": None,
        "init_from_gps": False,
        "add_map_mask": False,
        "mask_radius": None,
        "mask_pad": 1,
        "return_gps": False,
        # point cloud
        "select_num_points": None,
        # image preprocessing
        "target_focal_length": None,
        "resize_image": None,
        "pad_to_square": False,  # legacy
        "pad_to_multiple": 32,
        "rectify_pitch": True,
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

    def __init__(
        self,
        stage: str,
        cfg: DictConfig,
        names: List[str],
        data: Dict[str, Any],
        image_dirs: Dict[str, Path],
        tile_managers: Dict[str, TileManager],
        image_ext: str = "",
    ):
        self.stage = stage
        self.cfg = deepcopy(cfg)
        self.data = data
        self.image_dirs = image_dirs
        self.tile_managers = tile_managers
        self.names = names
        self.image_ext = image_ext

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
        if self.stage == "train" and self.cfg.random:
            seed = None
        else:
            seed = [self.cfg.seed, idx]
        (seed,) = np.random.SeedSequence(seed).generate_state(1)

        scene, seq, name = self.names[idx]
        if self.cfg.init_from_gps:
            latlon_gps = self.data["gps_position"][idx][:2].clone().numpy()
            xy_w_init = self.tile_managers[scene].projection.project(latlon_gps)
        else:
            xy_w_init = self.data["t_c2w"][idx][:2].clone().double().numpy()

        if "shifts" in self.data:
            yaw = self.data["roll_pitch_yaw"][idx][-1]
            R_c2w = rotmat2d((90 - yaw) / 180 * np.pi).float()
            error = (R_c2w @ self.data["shifts"][idx][:2]).numpy()
        else:
            error = np.random.RandomState(seed).uniform(-1, 1, size=2)
        xy_w_init += error * self.cfg.max_init_error

        bbox_tile = BoundaryBox(
            xy_w_init - self.cfg.crop_size_meters,
            xy_w_init + self.cfg.crop_size_meters,
        )
        return self.get_view(idx, scene, seq, name, seed, bbox_tile)

    def get_view(self, idx, scene, seq, name, seed, bbox_tile):
        data = {
            "index": idx,
            "name": name,
            "scene": scene,
            "sequence": seq,
        }
        cam_dict = self.data["cameras"][scene][seq][self.data["camera_id"][idx]]
        cam = Camera.from_colmap(cam_dict).float()

        if "roll_pitch_yaw" in self.data:
            roll, pitch, yaw = self.data["roll_pitch_yaw"][idx].numpy()
        else:
            roll, pitch, yaw = decompose_rotmat(self.data["R_c2w"][idx].numpy())
        image = read_image(self.image_dirs[scene] / (name + self.image_ext))

        if "plane_params" in self.data:
            data["camera_height"] = self.data["height"][idx].clone()
            # transform the plane parameters from world to camera frames
            plane_w = self.data["plane_params"][idx]
            data["ground_plane"] = torch.cat(
                [rotmat2d(deg2rad(torch.tensor(yaw))) @ plane_w[:2], plane_w[2:]]
            )

        # raster extraction
        canvas = self.tile_managers[scene].query(bbox_tile)
        xy_w_gt = self.data["t_c2w"][idx][:2].numpy()
        uv_gt = canvas.to_uv(xy_w_gt)
        uv_init = canvas.to_uv(bbox_tile.center)
        raster = canvas.raster  # C, H, W

        # Map augmentations
        # TODO: handle K
        heading = np.deg2rad(90 - yaw)  # fixme
        if self.stage == "train":
            if self.cfg.augmentation.rot90:
                raster, uv_gt, heading = random_rot90(raster, uv_gt, heading, seed)
            if self.cfg.augmentation.flip:
                image, raster, uv_gt, heading = random_flip(
                    image, raster, uv_gt, heading, seed
                )
        yaw = 90 - np.rad2deg(heading)  # fixme

        image, valid, cam, roll, pitch = self.process_image(
            image, cam, roll, pitch, seed
        )

        # Create the mask for prior location
        if self.cfg.add_map_mask:
            data["map_mask"] = torch.from_numpy(self.create_map_mask(canvas))

        if self.cfg.max_init_error_rotation is not None and "shifts" in self.data:
            yaw_error = self.data["shifts"][idx][-1] * self.cfg.max_init_error_rotation
            data["yaw_init"] = yaw + yaw_error

        if self.cfg.select_num_points is not None:
            data["pointcloud"], data["R_rect2cam"] = self.get_pointcloud(
                scene, seq, idx, roll, pitch, yaw
            )

        if self.cfg.return_gps:
            gps = self.data["gps_position"][idx][:2].numpy()
            xy_gps = self.tile_managers[scene].projection.project(gps)
            data["xy_gps"] = torch.from_numpy(canvas.to_uv(xy_gps)).float()
            data["accuracy_gps"] = (
                self.data["gps_accuracy"][idx].clone()
                if "gps_accuracy" in self.data
                else torch.tensor(15.0)
            )

        if "chunk_index" in self.data:
            data["chunk_id"] = (scene, seq, self.data["chunk_index"][idx])

        return {
            **data,
            "image": image,
            "valid": valid,
            "camera": cam,
            "canvas": canvas,
            "map": torch.from_numpy(np.ascontiguousarray(raster)).long(),
            "xy": torch.from_numpy(uv_gt).float(),  # TODO: maybe rename to uv?
            "xy_init": torch.from_numpy(uv_init).float(),  # TODO: maybe rename to uv?
            "roll_pitch_yaw": torch.tensor((roll, pitch, yaw)).float(),
            "pixels_per_meter": torch.tensor(canvas.ppm).float(),
        }

    def process_image(self, image, cam, roll, pitch, seed):
        valid = None
        if self.cfg.target_focal_length is not None:
            # rectify before downsampling to minimize the blur
            image = (
                torch.from_numpy(np.ascontiguousarray(image))
                .float()
                .permute(2, 0, 1)
                .div_(255)
            )
            image, valid = rectify_image(
                image, cam, roll, pitch if self.cfg.rectify_pitch else None
            )
            image = image.mul_(255).permute(1, 2, 0).numpy()
            valid = valid.numpy()
            roll = 0.0
            if self.cfg.rectify_pitch:
                pitch = 0.0
            # resize to a canonical focal length
            factor = self.cfg.target_focal_length / cam.f.numpy()
            size = (np.array(image.shape[:2][::-1]) * factor).astype(int)
            image, _, cam, valid = resize_image(image, size, camera=cam, valid=valid)
            size_out = self.cfg.resize_image
            if size_out is None:
                # round the edges up such that they are multiple of a factor
                stride = self.cfg.pad_to_multiple
                size_out = (np.ceil((size / stride)) * stride).astype(int)
            # crop or pad such that both edges are of the given size
            image, valid, cam = center_pad_crop_image(image, size_out, cam, valid)
            image /= 255
            np.clip(image, 0, 1, out=image)
        elif self.cfg.resize_image is not None:
            # legacy branch: resize to a fixed size and rectify after resizing
            # resize such that the longest edge has a given size
            image, _, cam = resize_image(
                image, self.cfg.resize_image, fn=max, camera=cam
            )
            if self.cfg.pad_to_square:
                # pad such that both edges are of the given size
                image, valid, cam = pad_image(image, self.cfg.resize_image, cam)

        # Convert to torch
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            image = self.tfs(np.ascontiguousarray(image))
        if valid is not None:
            valid = torch.from_numpy(valid)

        # legacy
        if self.cfg.target_focal_length is None:
            image, valid = rectify_image(
                image, cam, roll, pitch if self.cfg.rectify_pitch else None, valid
            )
            roll = 0.0
            if self.cfg.rectify_pitch:
                pitch = 0.0

        return image, valid, cam, roll, pitch

    def create_map_mask(self, canvas):
        map_mask = np.zeros(canvas.raster.shape[-2:], np.bool)
        radius = self.cfg.mask_radius or self.cfg.max_init_error
        mask_min, mask_max = np.round(
            canvas.to_uv(canvas.bbox.center)
            + np.array([[-1], [1]]) * (radius + self.cfg.mask_pad) * canvas.ppm
        ).astype(int)
        map_mask[mask_min[1] : mask_max[1], mask_min[0] : mask_max[0]] = True
        return map_mask

    def get_pointcloud(self, scene, seq, idx, roll, pitch, yaw):
        points = self.data["points"][scene][seq][str(self.data["chunk_id"][idx])]
        points = points[self.data["observations"][idx]]
        if self.cfg.select_num_points != -1:
            # TODO: select randomly with seed
            points = points[: self.cfg.select_num_points]
        points = (points - self.dataview["t_c2w"][idx]).float()
        points = points @ Rotation.from_euler(
            "Z", -yaw, degrees=True
        ).as_matrix().astype(np.float32)
        R_rect2cam = Rotation.from_euler("ZX", [roll, pitch + 90], degrees=True)
        return points, R_rect2cam.as_matrix().astype(np.float32)
