# Copyright (c) Meta Platforms, Inc. and affiliates.

# Adapted from PixLoc, Paul-Edouard Sarlin, ETH Zurich
# https://github.com/cvg/pixloc
# Released under the Apache License 2.0

"""
Convenience classes for an SE3 pose and a pinhole Camera with lens distortion.
Based on PyTorch tensors: differentiable, batched, with GPU support.
"""

import functools
import inspect
from typing import Dict, List, NamedTuple, Tuple, Union

import numpy as np
import torch

from .geometry import undistort_points


def autocast(func):
    """Cast the inputs of a TensorWrapper method to PyTorch tensors
    if they are numpy arrays. Use the device and dtype of the wrapper.
    """

    @functools.wraps(func)
    def wrap(self, *args):
        device = torch.device("cpu")
        dtype = None
        if isinstance(self, TensorWrapper):
            if self._data is not None:
                device = self.device
                dtype = self.dtype
        elif not inspect.isclass(self) or not issubclass(self, TensorWrapper):
            raise ValueError(self)

        cast_args = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                arg = torch.from_numpy(arg)
                arg = arg.to(device=device, dtype=dtype)
            cast_args.append(arg)
        return func(self, *cast_args)

    return wrap


class TensorWrapper:
    _data = None

    @autocast
    def __init__(self, data: torch.Tensor):
        self._data = data

    @property
    def shape(self):
        return self._data.shape[:-1]

    @property
    def device(self):
        return self._data.device

    @property
    def dtype(self):
        return self._data.dtype

    def __getitem__(self, index):
        return self.__class__(self._data[index])

    def __setitem__(self, index, item):
        self._data[index] = item.data

    def to(self, *args, **kwargs):
        return self.__class__(self._data.to(*args, **kwargs))

    def cpu(self):
        return self.__class__(self._data.cpu())

    def cuda(self):
        return self.__class__(self._data.cuda())

    def pin_memory(self):
        return self.__class__(self._data.pin_memory())

    def float(self):
        return self.__class__(self._data.float())

    def double(self):
        return self.__class__(self._data.double())

    def detach(self):
        return self.__class__(self._data.detach())

    @classmethod
    def stack(cls, objects: List, dim=0, *, out=None):
        data = torch.stack([obj._data for obj in objects], dim=dim, out=out)
        return cls(data)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func is torch.stack:
            return cls.stack(*args, **kwargs)
        else:
            return NotImplemented


class Transform3D(TensorWrapper):
    """SE(3) transformation with 3-DoF translation and 3-DoF rotation"""

    def __init__(self, data: torch.Tensor):
        assert data.shape[-1] == 12
        super().__init__(data)

    @classmethod
    @autocast
    def from_Rt(cls, R: torch.Tensor, t: torch.Tensor):
        """Pose from a rotation matrix and translation vector.
        Accepts numpy arrays or PyTorch tensors.

        Args:
            R: rotation matrix with shape (..., 3, 3).
            t: translation vector with shape (..., 3).
        """
        assert R.shape[-2:] == (3, 3)
        assert t.shape[-1] == 3
        assert R.shape[:-2] == t.shape[:-1]
        data = torch.cat([R.flatten(start_dim=-2), t], -1)
        return cls(data)

    @classmethod
    def from_4x4mat(cls, T: torch.Tensor):
        """Pose from an SE(3) transformation matrix.
        Args:
            T: transformation matrix with shape (..., 4, 4).
        """
        assert T.shape[-2:] == (4, 4)
        R, t = T[..., :3, :3], T[..., :3, 3]
        return cls.from_Rt(R, t)

    @classmethod
    def from_colmap(cls, image: NamedTuple):
        """Pose from a COLMAP Image."""
        return cls.from_Rt(image.qvec2rotmat(), image.tvec)

    @property
    def R(self) -> torch.Tensor:
        """Underlying rotation matrix with shape (..., 3, 3)."""
        rvec = self._data[..., :9]
        return rvec.reshape(rvec.shape[:-1] + (3, 3))

    @property
    def t(self) -> torch.Tensor:
        """Underlying translation vector with shape (..., 3)."""
        return self._data[..., -3:]

    def inv(self) -> "Transform3D":
        """Invert an SE(3) pose."""
        R = self.R.transpose(-1, -2)
        t = -(R @ self.t.unsqueeze(-1)).squeeze(-1)
        return self.__class__.from_Rt(R, t)

    def compose(self, other: "Transform3D") -> "Transform3D":
        """Chain two SE(3) poses: C_T_B.compose(B_T_A) - > C_T_A"""
        R = self.R @ other.R
        t = self.t + (self.R @ other.t.unsqueeze(-1)).squeeze(-1)
        return self.__class__.from_Rt(R, t)

    @autocast
    def transform(self, p3d: torch.Tensor) -> torch.Tensor:
        """Transform a set of 3D points.
        Args:
            p3d: 3D points, numpy array or PyTorch tensor with shape (..., 3).
        """
        assert p3d.shape[-1] == 3
        return p3d @ self.R.transpose(-1, -2) + self.t.unsqueeze(-2)

    def __matmul__(
        self, other: Union["Transform3D", torch.Tensor]
    ) -> Union["Transform3D", torch.Tensor]:
        """Transform a set of 3D points: B_T_A @ p3d_A -> p3D_B
        or chain two SE(3) poses: C_T_B @ B_T_A -> C_T_A"""
        if isinstance(other, self.__class__):
            return self.compose(other)
        else:
            return self.transform(other)

    def numpy(self) -> Tuple[np.ndarray]:
        return self.R.numpy(), self.t.numpy()

    def magnitude(self) -> Tuple[torch.Tensor]:
        """Magnitude of the SE(3) transformation.
        Returns:
            dr: rotation angle in degrees.
            dt: translation distance in meters.
        """
        trace = torch.diagonal(self.R, dim1=-1, dim2=-2).sum(-1)
        cos = torch.clamp((trace - 1) / 2, -1, 1)
        dr = torch.rad2deg(torch.acos(cos).abs())
        dt = torch.norm(self.t, dim=-1)
        return dr, dt

    def __repr__(self):
        return f"Transform3D: {self.shape} {self.dtype} {self.device}"


class Transform2D(TensorWrapper):
    """SE(2) transformation with 2-DoF translation and 1-DoF rotation"""

    def __init__(self, data: torch.Tensor):
        assert data.shape[-1] == 3  # angle_deg, x, y
        super().__init__(data)

    @property
    def angle(self) -> torch.Tensor:
        """Returns angle in degrees"""
        return self._data[..., :1]

    @property
    def t(self) -> torch.Tensor:
        """Underlying translation vector with shape (..., 2)."""
        return self._data[..., -2:]

    @classmethod
    def from_degrees(cls, angle: torch.Tensor, t: torch.Tensor):
        """SE(2) pose from degrees. Rotation is counter-clockwise from +ve X"""
        rt_flat = torch.cat([angle, t[..., 0:2]], -1)
        return cls(rt_flat)

    @classmethod
    @autocast
    def from_Rt(cls, R: torch.Tensor, t: torch.Tensor):
        """Pose from a 2D rotation matrix and 2D translation vector.
        Accepts numpy arrays or PyTorch tensors.

        Args:
            R: rotation matrix with shape (..., 2, 2).
            t: translation vector with shape (..., 2).
        """
        assert R.shape[-2:] == (2, 2)
        assert t.shape[-1] == 2
        assert R.shape[:-2] == t.shape[:-1]
        angle_deg = torch.rad2deg(torch.arctan2(R[..., 1, 0], R[..., 0, 0]))
        return cls.from_degrees(angle_deg[None], t[..., :2])

    @classmethod
    def camera_2d_from_3d(cls, transform: "Transform3D"):
        """SE(2) pose from an SE(3) pose.
        Computes yaw as angle between x-axis and camera's z-axis."""
        angle_deg = torch.rad2deg(
            torch.arctan2(transform.R[..., 1, 2][None], transform.R[..., 0, 2][None])
        )
        return cls.from_degrees(angle_deg, transform.t[..., :2])

    @classmethod
    def from_Transform3D(cls, transform: "Transform3D"):
        """SE(2) pose from an SE(3) pose."""
        angle_deg = torch.rad2deg(
            torch.arctan2(transform.R[..., 1, 0][None], transform.R[..., 0, 0][None])
        )
        return cls.from_degrees(angle_deg, transform.t[..., :2])

    @property
    def R(self) -> torch.Tensor:
        """Underlying rotation matrix with shape (..., 2, 2)."""
        rad = torch.deg2rad(self.angle)
        cos = torch.cos(rad)
        sin = torch.sin(rad)
        R_flat = torch.cat([cos, -sin, sin, cos], -1)
        return R_flat.reshape(R_flat.shape[:-1] + (2, 2))

    def inv(self) -> "Transform2D":
        """Invert an SE(2) pose."""
        R = self.R.transpose(-1, -2)
        t = -(R @ self.t.unsqueeze(-1)).squeeze(-1)
        return self.__class__.from_Rt(R, t)

    def compose(self, other: "Transform2D") -> "Transform2D":
        """Chain two SE(2) poses: C_T_B.compose(B_T_A) -> C_T_A."""
        R = self.R @ other.R
        t = self.t + (self.R @ other.t.unsqueeze(-1)).squeeze(-1)
        return self.__class__.from_Rt(R, t)

    @autocast
    def transform(self, p2d: torch.Tensor) -> torch.Tensor:
        """Transform a set of 2D points.
        Args:
            p2d: 2D points, numpy array or PyTorch tensor with shape (..., 2).
        """
        assert p2d.shape[-1] == 2
        # assert p3d.shape[:-2] == self.shape  # allow broadcasting
        return p2d @ self.R.transpose(-1, -2) + self.t.unsqueeze(-2)

    def __matmul__(
        self, other: Union["Transform2D", torch.Tensor]
    ) -> Union["Transform2D", torch.Tensor]:
        """Transform a set of 2D points: B_T_A * p2D_A -> p2D_B.
        or chain two SE(2) poses: C_T_B @ B_T_A -> C_T_A."""
        if isinstance(other, self.__class__):
            return self.compose(other)
        else:
            return self.transform(other)

    def numpy(self) -> Tuple[np.ndarray]:
        return self.R.numpy(), self.t.numpy()

    def magnitude(self) -> Tuple[torch.Tensor]:
        """Magnitude of the SE(2) transformation.
        Returns:
            dr: rotation angle in degrees.
            dt: translation distance in meters.
        """
        dr = self.angle % 360
        dr = torch.min(dr, 360 - dr)
        dt = torch.norm(self.t, dim=-1)
        return dr, dt

    def __repr__(self):
        return f"Transform2D: {self.shape} {self.dtype} {self.device}. \
            angle: {self.angle}. t: {self.t}"


class Camera(TensorWrapper):
    eps = 1e-4

    def __init__(self, data: torch.Tensor):
        assert data.shape[-1] in {6, 8, 10}
        super().__init__(data)

    @classmethod
    def from_dict(cls, camera: Union[Dict, NamedTuple]):
        """Camera from a COLMAP Camera tuple or dictionary.
        We assume that the origin (0, 0) is the center of the top-left pixel.
        This is different from COLMAP.
        """
        if isinstance(camera, tuple):
            camera = camera._asdict()

        model = camera["model"]
        params = camera["params"]

        if model in ["OPENCV", "PINHOLE"]:
            (fx, fy, cx, cy), params = np.split(params, [4])
        elif model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"]:
            (f, cx, cy), params = np.split(params, [3])
            fx = fy = f
            if model == "SIMPLE_RADIAL":
                params = np.r_[params, 0.0]
        else:
            raise NotImplementedError(model)

        data = np.r_[
            camera["width"], camera["height"], fx, fy, cx - 0.5, cy - 0.5, params
        ]
        return cls(data)

    @property
    def size(self) -> torch.Tensor:
        """Size (width height) of the images, with shape (..., 2)."""
        return self._data[..., :2]

    @property
    def f(self) -> torch.Tensor:
        """Focal lengths (fx, fy) with shape (..., 2)."""
        return self._data[..., 2:4]

    @property
    def c(self) -> torch.Tensor:
        """Principal points (cx, cy) with shape (..., 2)."""
        return self._data[..., 4:6]

    @property
    def dist(self) -> torch.Tensor:
        """Distortion parameters, with shape (..., {0, 2, 4})."""
        return self._data[..., 6:]

    def scale(self, scales: Union[float, int, Tuple[Union[float, int]]]):
        """Update the camera parameters after resizing an image."""
        if isinstance(scales, (int, float)):
            scales = (scales, scales)
        s = self._data.new_tensor(scales)
        data = torch.cat(
            [self.size * s, self.f * s, (self.c + 0.5) * s - 0.5, self.dist], -1
        )
        return self.__class__(data)

    def crop(self, left_top: Tuple[float], size: Tuple[int]):
        """Update the camera parameters after cropping an image."""
        left_top = self._data.new_tensor(left_top)
        size = self._data.new_tensor(size)
        data = torch.cat([size, self.f, self.c - left_top, self.dist], -1)
        return self.__class__(data)

    @autocast
    def in_image(self, p2d: torch.Tensor):
        """Check if 2D points are within the image boundaries."""
        assert p2d.shape[-1] == 2
        # assert p2d.shape[:-2] == self.shape  # allow broadcasting
        size = self.size.unsqueeze(-2)
        valid = torch.all((p2d >= 0) & (p2d <= (size - 1)), -1)
        return valid

    @autocast
    def project(self, p3d: torch.Tensor) -> Tuple[torch.Tensor]:
        """Project 3D points into the camera plane and check for visibility."""
        z = p3d[..., -1]
        valid = z > self.eps
        z = z.clamp(min=self.eps)
        p2d = p3d[..., :-1] / z.unsqueeze(-1)
        return p2d, valid

    def J_project(self, p3d: torch.Tensor):
        x, y, z = p3d[..., 0], p3d[..., 1], p3d[..., 2]
        zero = torch.zeros_like(z)
        J = torch.stack([1 / z, zero, -x / z**2, zero, 1 / z, -y / z**2], dim=-1)
        J = J.reshape(p3d.shape[:-1] + (2, 3))
        return J  # N x 2 x 3

    @autocast
    def undistort(self, pts: torch.Tensor) -> Tuple[torch.Tensor]:
        """Undistort normalized 2D coordinates
        and check for validity of the distortion model.
        """
        assert pts.shape[-1] == 2
        # assert pts.shape[:-2] == self.shape  # allow broadcasting
        return undistort_points(pts, self.dist)

    @autocast
    def denormalize(self, p2d: torch.Tensor) -> torch.Tensor:
        """Convert normalized 2D coordinates into pixel coordinates."""
        return p2d * self.f.unsqueeze(-2) + self.c.unsqueeze(-2)

    @autocast
    def normalize(self, p2d: torch.Tensor) -> torch.Tensor:
        """Convert pixel coordinates into normalized 2D coordinates."""
        return (p2d - self.c.unsqueeze(-2)) / self.f.unsqueeze(-2)

    def J_denormalize(self):
        return torch.diag_embed(self.f).unsqueeze(-3)  # 1 x 2 x 2

    @autocast
    def world2image(self, p3d: torch.Tensor) -> Tuple[torch.Tensor]:
        """Transform 3D points into 2D pixel coordinates."""
        p2d, visible = self.project(p3d)
        p2d, mask = self.undistort(p2d)
        p2d = self.denormalize(p2d)
        valid = visible & mask & self.in_image(p2d)
        return p2d, valid

    def J_world2image(self, p3d: torch.Tensor):
        p2d_dist, valid = self.project(p3d)
        J = self.J_denormalize() @ self.J_undistort(p2d_dist) @ self.J_project(p3d)
        return J, valid

    def __repr__(self):
        return f"Camera {self.shape} {self.dtype} {self.device}"
