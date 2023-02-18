# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import numpy as np
import torch


def from_homogeneous(points, eps: float = 1e-8):
    """Remove the homogeneous dimension of N-dimensional points.
    Args:
        points: torch.Tensor or numpy.ndarray with size (..., N+1).
    Returns:
        A torch.Tensor or numpy ndarray with size (..., N).
    """
    return points[..., :-1] / (points[..., -1:] + eps)


def to_homogeneous(points):
    """Convert N-dimensional points to homogeneous coordinates.
    Args:
        points: torch.Tensor or numpy.ndarray with size (..., N).
    Returns:
        A torch.Tensor or numpy.ndarray with size (..., N+1).
    """
    if isinstance(points, torch.Tensor):
        pad = points.new_ones(points.shape[:-1] + (1,))
        return torch.cat([points, pad], dim=-1)
    elif isinstance(points, np.ndarray):
        pad = np.ones((points.shape[:-1] + (1,)), dtype=points.dtype)
        return np.concatenate([points, pad], axis=-1)
    else:
        raise ValueError


@torch.jit.script
def undistort_points(pts, dist):
    """Undistort normalized 2D coordinates
    and check for validity of the distortion model.
    Adapted from PixLoc, Paul-Edouard Sarlin, ETH Zurich
    https://github.com/cvg/pixloc
    Released under the Apache License 2.0
    """
    dist = dist.unsqueeze(-2)  # add point dimension
    ndist = dist.shape[-1]
    undist = pts
    valid = torch.ones(pts.shape[:-1], device=pts.device, dtype=torch.bool)
    if ndist > 0:
        k1, k2 = dist[..., :2].split(1, -1)
        r2 = torch.sum(pts**2, -1, keepdim=True)
        radial = k1 * r2 + k2 * r2**2
        undist = undist + pts * radial

        # The distortion model is supposedly only valid within the image
        # boundaries. Because of the negative radial distortion, points that
        # are far outside of the boundaries might actually be mapped back
        # within the image. To account for this, we discard points that are
        # beyond the inflection point of the distortion model,
        # e.g. such that d(r + k_1 r^3 + k2 r^5)/dr = 0
        limited = ((k2 > 0) & ((9 * k1**2 - 20 * k2) > 0)) | ((k2 <= 0) & (k1 > 0))
        limit = torch.abs(
            torch.where(
                k2 > 0,
                (torch.sqrt(9 * k1**2 - 20 * k2) - 3 * k1) / (10 * k2),
                1 / (3 * k1),
            )
        )
        valid = valid & torch.squeeze(~limited | (r2 < limit), -1)

        if ndist > 2:
            p12 = dist[..., 2:]
            p21 = p12.flip(-1)
            uv = torch.prod(pts, -1, keepdim=True)
            undist = undist + 2 * p12 * uv + p21 * (r2 + 2 * pts**2)
            # TODO: handle tangential boundaries

    return undist, valid
