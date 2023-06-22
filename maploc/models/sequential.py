# Copyright (c) Meta Platforms, Inc. and affiliates.

import numpy as np
import torch

from .voting import argmax_xyr, log_softmax_spatial, sample_xyr
from .utils import deg2rad, make_grid, rotmat2d


def log_gaussian(points, mean, sigma):
    return -1 / 2 * torch.sum((points - mean) ** 2, -1) / sigma**2


def log_laplace(points, mean, sigma):
    return -torch.sum(torch.abs(points - mean), -1) / sigma


def propagate_belief(
    Δ_xy, Δ_yaw, canvas_target, canvas_source, belief, num_rotations=None
):
    # We allow a different sampling resolution in the target frame
    if num_rotations is None:
        num_rotations = belief.shape[-1]

    angles = torch.arange(
        0, 360, 360 / num_rotations, device=Δ_xy.device, dtype=Δ_xy.dtype
    )
    uv_grid = make_grid(canvas_target.w, canvas_target.h, device=Δ_xy.device)
    xy_grid = canvas_target.to_xy(uv_grid.to(Δ_xy))

    Δ_xy_world = torch.einsum("nij,j->ni", rotmat2d(deg2rad(-angles)), Δ_xy)
    xy_grid_prev = xy_grid[..., None, :] + Δ_xy_world[..., None, None, :, :]
    uv_grid_prev = canvas_source.to_uv(xy_grid_prev).to(Δ_xy)

    angles_prev = angles + Δ_yaw
    angles_grid_prev = angles_prev.tile((canvas_target.h, canvas_target.w, 1))

    prior, valid = sample_xyr(
        belief[None, None],
        uv_grid_prev.to(belief)[None],
        angles_grid_prev.to(belief)[None],
        nearest_for_inf=True,
    )
    return prior, valid


def markov_filtering(observations, canvas, xys, yaws, idxs=None):
    assert len(observations) == len(canvas) == len(xys) == len(yaws)
    if idxs is None:
        idxs = range(len(observations))
    belief = None
    beliefs = []
    for i in idxs:
        obs = observations[i]
        if belief is None:
            belief = obs
        else:
            Δ_xy = rotmat2d(deg2rad(yaws[i])) @ (xys[i - 1] - xys[i])
            Δ_yaw = yaws[i - 1] - yaws[i]
            prior, valid = propagate_belief(
                Δ_xy, Δ_yaw, canvas[i], canvas[i - 1], belief
            )
            prior = prior[0, 0].masked_fill_(~valid[0], -np.inf)
            belief = prior + obs
            belief = log_softmax_spatial(belief)
        beliefs.append(belief)
    uvt_seq = torch.stack([argmax_xyr(p) for p in beliefs])
    return beliefs, uvt_seq


def integrate_observation(
    source,
    target,
    xy_source,
    xy_target,
    yaw_source,
    yaw_target,
    canvas_source,
    canvas_target,
    **kwargs
):
    Δ_xy = rotmat2d(deg2rad(yaw_target)) @ (xy_source - xy_target)
    Δ_yaw = yaw_source - yaw_target
    prior, valid = propagate_belief(
        Δ_xy, Δ_yaw, canvas_target, canvas_source, source, **kwargs
    )
    prior = prior[0, 0].masked_fill_(~valid[0], -np.inf)
    target.add_(prior)
    target.sub_(target.max())  # normalize to avoid overflow
    return prior


class RigidAligner:
    def __init__(
        self,
        canvas_ref=None,
        xy_ref=None,
        yaw_ref=None,
        num_rotations=None,
        track_priors=False,
    ):
        self.canvas = canvas_ref
        self.xy_ref = xy_ref
        self.yaw_ref = yaw_ref
        self.rotmat_ref = None
        self.num_rotations = num_rotations
        self.belief = None
        self.priors = [] if track_priors else None

        self.yaw_slam2geo = None
        self.Rt_slam2geo = None

    def update(self, observation, canvas, xy, yaw):
        # initialization
        if self.canvas is None:
            self.canvas = canvas
        if self.xy_ref is None:
            self.xy_ref = xy
            self.yaw_ref = yaw
            self.rotmat_ref = rotmat2d(deg2rad(self.yaw_ref))
        if self.num_rotations is None:
            self.num_rotations = observation.shape[-1]
        if self.belief is None:
            self.belief = observation.new_zeros(
                (self.canvas.h, self.canvas.w, self.num_rotations)
            )

        prior = integrate_observation(
            observation,
            self.belief,
            xy,
            self.xy_ref,
            yaw,
            self.yaw_ref,
            canvas,
            self.canvas,
            num_rotations=self.num_rotations,
        )

        if self.priors is not None:
            self.priors.append(prior.cpu())
        return prior

    def update_with_ref(self, observation, canvas, xy, yaw):
        if self.belief is not None:
            observation = observation.clone()
            integrate_observation(
                self.belief,
                observation,
                self.xy_ref,
                xy,
                self.yaw_ref,
                yaw,
                self.canvas,
                canvas,
                num_rotations=observation.shape[-1],
            )

        self.belief = observation
        self.canvas = canvas
        self.xy_ref = xy
        self.yaw_ref = yaw

    def compute(self):
        uvt_align_ref = argmax_xyr(self.belief)
        self.yaw_ref_align = uvt_align_ref[-1]
        self.xy_ref_align = self.canvas.to_xy(uvt_align_ref[:2].double())

        self.yaw_slam2geo = self.yaw_ref - self.yaw_ref_align
        R_slam2geo = rotmat2d(deg2rad(self.yaw_slam2geo))
        t_slam2geo = self.xy_ref_align - R_slam2geo @ self.xy_ref
        self.Rt_slam2geo = (R_slam2geo, t_slam2geo)

    def transform(self, xy, yaw):
        if self.Rt_slam2geo is None or self.yaw_slam2geo is None:
            raise ValueError("Missing transformation, call `compute()` first!")
        xy_geo = self.Rt_slam2geo[1].to(xy) + xy @ self.Rt_slam2geo[0].T.to(xy)
        return xy_geo, (self.yaw_slam2geo.to(yaw) + yaw) % 360


class GPSAligner(RigidAligner):
    def __init__(self, distribution=log_laplace, **kwargs):
        self.distribution = distribution
        super().__init__(**kwargs)
        if self.num_rotations is None:
            raise ValueError("Rotation number is required.")
        angles = torch.arange(0, 360, 360 / self.num_rotations)
        self.rotmats = rotmat2d(deg2rad(-angles))
        self.xy_grid = None

    def update(self, xy_gps, accuracy, canvas, xy, yaw):
        # initialization
        if self.canvas is None:
            self.canvas = canvas
        if self.xy_ref is None:
            self.xy_ref = xy
            self.yaw_ref = yaw
            self.rotmat_ref = rotmat2d(deg2rad(self.yaw_ref))
        if self.xy_grid is None:
            self.xy_grid = self.canvas.to_xy(make_grid(self.canvas.w, self.canvas.h))
        if self.belief is None:
            self.belief = xy_gps.new_zeros(
                (self.canvas.h, self.canvas.w, self.num_rotations)
            )

        # integration
        Δ_xy = self.rotmat_ref @ (xy - self.xy_ref)
        Δ_xy_world = torch.einsum("nij,j->ni", self.rotmats.to(xy), Δ_xy)
        xy_grid_prev = (
            self.xy_grid.to(xy)[..., None, :] + Δ_xy_world[..., None, None, :, :]
        )
        prior = self.distribution(xy_grid_prev, xy_gps, accuracy)
        self.belief.add_(prior)
        self.belief.sub_(self.belief.max())  # normalize to avoid overflow

        if self.priors is not None:
            self.priors.append(prior.cpu())
        return prior
