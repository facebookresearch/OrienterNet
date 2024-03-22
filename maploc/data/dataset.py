# Copyright (c) Meta Platforms, Inc. and affiliates.

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.utils.data as torchdata
import torchvision.transforms as tvf
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.transform import Rotation

from ..models.utils import rotmat2d
from ..osm.tiling import TileManager
from ..utils.geo import BoundaryBox
from ..utils.io import read_image
from ..utils.wrappers import Camera, Transform2D, Transform3D
from .image import pad_image, rectify_image, resize_image
from .utils import compose_rotmat, random_flip, random_rot90


class MapLocDataset(torchdata.Dataset):
    default_cfg = {
        "seed": 0,
        "accuracy_gps": 15,
        "random": True,
        "num_threads": None,
        # map
        "num_classes": None,
        "pixel_per_meter": "???",
        "crop_size_meters": "???",
        "max_init_error": "???",
        "max_init_error_rotation": None,
        "init_from_gps": False,
        "return_gps": False,
        "force_camera_height": None,
        # pose priors
        "add_map_mask": False,
        "mask_radius": None,
        "mask_pad": 1,
        "prior_range_rotation": None,
        # image preprocessing
        "target_focal_length": None,
        "reduce_fov": None,
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

        tfs = []
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
            world_R_cam = rotmat2d(np.deg2rad((90 - yaw))).float()
            error = (world_R_cam @ self.data["shifts"][idx][:2]).numpy()
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
        cam = Camera.from_dict(cam_dict).float()

        # for backward compatibility
        if "roll_pitch_yaw" in self.data:
            world_R_cam = compose_rotmat(*self.data["roll_pitch_yaw"][idx].numpy())
        else:
            world_R_cam = self.data["R_c2w"][idx].numpy()

        world_t_cam = self.data["t_c2w"][idx].numpy()

        image = read_image(self.image_dirs[scene] / (name + self.image_ext))

        if self.cfg.force_camera_height is not None:
            data["camera_height"] = torch.tensor(self.cfg.force_camera_height)
        elif "camera_height" in self.data:
            data["camera_height"] = self.data["height"][idx].clone()

        # raster extraction
        canvas = self.tile_managers[scene].query(bbox_tile)
        raster = canvas.raster  # C, H, W
        raster = torch.from_numpy(np.ascontiguousarray(raster)).long()
        world_T_cam = Transform3D.from_Rt(world_R_cam, world_t_cam).float()
        world_T_cam2d = Transform2D.camera_2d_from_3d(world_T_cam)

        # gcam: gravity-aligned camera with z=optical axis
        # gcamxyz: gcam rotated such that z:up,x:right,y:forward.

        gcam_angle = world_T_cam2d.angle - 90
        Rz = Transform2D.from_degrees(gcam_angle, torch.zeros(2))
        world_R_gcamxyz = torch.eye(3)
        world_R_gcamxyz[:2, :2] = Rz.R
        world_T_gcamxyz = Transform3D.from_Rt(world_R_gcamxyz, world_T_cam.t).float()
        gcamxyz_T_gcam = Transform3D.from_Rt(
            Rotation.from_euler("X", -90, degrees=True).as_matrix(), torch.zeros(3)
        ).float()
        world_T_gcam = world_T_gcamxyz @ gcamxyz_T_gcam
        gcam_T_cam = world_T_gcam.inv() @ world_T_cam
        cam_R_gcam = gcam_T_cam.inv().R

        world_T_tile = Transform2D.from_Rt(torch.eye(2), canvas.bbox.min_).float()
        tile_T_cam = world_T_tile.inv() @ world_T_cam2d

        image, valid, cam = self.process_image(image, cam, seed, cam_R_gcam)

        # Map augmentations
        if self.stage == "train":
            if self.cfg.augmentation.rot90:
                raster, tile_T_cam = random_rot90(raster, tile_T_cam, canvas.ppm)
            if self.cfg.augmentation.flip:
                image, valid, raster, tile_T_cam = random_flip(
                    image, valid, raster, tile_T_cam, canvas.ppm
                )
        map_T_cam = Transform2D.to_pixels(tile_T_cam, canvas.ppm)
        # map_T_cam will be deprecated, tile_T_cam is sufficient.

        world_t_init = torch.from_numpy(bbox_tile.center).float()
        tile_t_init = world_t_init - world_T_tile.t
        map_t_init = Transform2D.to_pixels(tile_t_init, canvas.ppm)

        # Create the mask for prior location
        if self.cfg.add_map_mask:
            data["map_mask"] = torch.from_numpy(self.create_map_mask(canvas))

        if self.cfg.max_init_error_rotation is not None:
            if "shifts" in self.data:
                error = self.data["shifts"][idx][-1]
            else:
                error = np.random.RandomState(seed + 1).uniform(-1, 1)
                error = torch.tensor(error, dtype=torch.float)
            yaw_init = tile_T_cam.angle + error * self.cfg.max_init_error_rotation
            range_ = self.cfg.prior_range_rotation or self.cfg.max_init_error_rotation
            data["yaw_prior"] = torch.stack([yaw_init, torch.tensor(range_)])

        if self.cfg.return_gps:
            gps = self.data["gps_position"][idx][:2].numpy()
            world_t_gps = self.tile_managers[scene].projection.project(gps)
            world_t_gps = torch.from_numpy(world_t_gps).float()
            tile_t_gps = world_t_gps - world_T_tile.t
            data["map_t_gps"] = Transform2D.to_pixels(tile_t_gps, canvas.ppm)
            data["accuracy_gps"] = torch.tensor(
                min(self.cfg.accuracy_gps, self.cfg.crop_size_meters)
            )

        if "chunk_index" in self.data:
            data["chunk_id"] = (scene, seq, self.data["chunk_index"][idx])

        return {
            **data,
            "image": image,
            "valid": valid,
            "camera": cam,
            "canvas": canvas,
            "map": raster,
            "tile_T_cam": tile_T_cam.float(),
            "map_T_cam": map_T_cam.float(),
            "map_t_init": map_t_init,
            "pixels_per_meter": torch.tensor(canvas.ppm).float(),
        }

    def process_image(self, image, cam, seed, cam_R_gcam):
        image = (
            torch.from_numpy(np.ascontiguousarray(image))
            .permute(2, 0, 1)
            .float()
            .div_(255)
        )
        assert self.cfg.rectify_pitch
        image, valid = rectify_image(image, cam, cam_R_gcam)

        if self.cfg.target_focal_length is not None:
            # resize to a canonical focal length
            factor = self.cfg.target_focal_length / cam.f.numpy()
            size = (np.array(image.shape[-2:][::-1]) * factor).astype(int)
            image, _, cam, valid = resize_image(image, size, camera=cam, valid=valid)
            size_out = self.cfg.resize_image
            if size_out is None:
                # round the edges up such that they are multiple of a factor
                stride = self.cfg.pad_to_multiple
                size_out = (np.ceil((size / stride)) * stride).astype(int)
            # crop or pad such that both edges are of the given size
            image, valid, cam = pad_image(
                image, size_out, cam, valid, crop_and_center=True
            )
        elif self.cfg.resize_image is not None:
            image, _, cam, valid = resize_image(
                image, self.cfg.resize_image, fn=max, camera=cam, valid=valid
            )
            if self.cfg.pad_to_square:
                # pad such that both edges are of the given size
                image, valid, cam = pad_image(image, self.cfg.resize_image, cam, valid)

        if self.cfg.reduce_fov is not None:
            h, w = image.shape[-2:]
            f = float(cam.f[0])
            fov = np.arctan(w / f / 2)
            w_new = round(2 * f * np.tan(self.cfg.reduce_fov * fov))
            image, valid, cam = pad_image(
                image, (w_new, h), cam, valid, crop_and_center=True
            )

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            image = self.tfs(image)
        return image, valid, cam

    def create_map_mask(self, canvas):
        map_mask = np.zeros(canvas.raster.shape[-2:], bool)
        radius = self.cfg.mask_radius or self.cfg.max_init_error
        mask_min, mask_max = np.round(
            canvas.to_uv(canvas.bbox.center)
            + np.array([[-1], [1]]) * (radius + self.cfg.mask_pad) * canvas.ppm
        ).astype(int)
        map_mask[mask_min[1] : mask_max[1], mask_min[0] : mask_max[0]] = True
        return map_mask
