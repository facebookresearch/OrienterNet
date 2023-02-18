# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import logging
from typing import Tuple

import torch
from kornia.filters.sobel import spatial_gradient3d

from .interpolation import Interpolator
from .utils import deg2rad, rad2deg, rotmat2d, rotmat2d_grad

logger = logging.getLogger(__name__)


class FeaturemetricRefiner:
    def __init__(
        self, grid_xy_cam, ppm, num_iters=30, interpolation="linear", damping_init=1e-4
    ):
        self.interpolator = Interpolator(mode=interpolation, pad=1)
        self.num_iters = num_iters
        self.damping_init = damping_init
        self.grid_uv_cam = grid_xy_cam * grid_xy_cam.new_tensor([1, -1]) * ppm
        self.grid_uv_cam_flat = self.grid_uv_cam.reshape(-1, 2)
        self.costs = []

    def __call__(self, xy_init, yaw_init, feats_map, feats_q, confidence, valid_bev):
        xy, yaw = xy_init, deg2rad(yaw_init)
        confidence = confidence.flatten(-2)
        feats_q = feats_q.flatten(-2).transpose(-1, -2)
        valid_bev = valid_bev.flatten(-2)
        damping = self.damping_init
        self.costs = []

        for i in range(self.num_iters):
            res, valid_proj, J_f_p2d = compute_residuals(
                xy,
                yaw,
                feats_map,
                feats_q,
                self.grid_uv_cam_flat,
                self.interpolator,
                grads=True,
            )
            weights = confidence * (valid_proj & valid_bev).float()
            error = torch.sum(res**2, -1) * weights
            cost = torch.mean(error)
            self.costs.append(cost.item())

            J_p2d_T = torch.cat(
                [
                    torch.diag_embed(torch.ones_like(self.grid_uv_cam_flat)),
                    (self.grid_uv_cam_flat @ rotmat2d_grad(yaw).T).unsqueeze(-1),
                ],
                -1,
            )
            J = J_f_p2d @ J_p2d_T
            g, H = build_system(res, J, weights)
            delta = optimizer_step(g, H, lambda_=damping)

            xy_new = xy + delta[:2]
            yaw_new = yaw + delta[2]
            res, valid_proj, _ = compute_residuals(
                xy_new,
                yaw_new,
                feats_map,
                feats_q,
                self.grid_uv_cam_flat,
                self.interpolator,
            )
            weights = confidence * (valid_proj & valid_bev).float()
            cost_new = torch.mean(torch.sum(res**2, -1) * weights)

            if cost_new > self.costs[-1]:
                damping *= 2
            else:
                damping /= 2
                xy = xy_new
                yaw = yaw_new
            if damping > 1e4 or ((i % 5 == 0) and torch.all(torch.abs(delta) < 1e-3)):
                # print("exit at iteration", i, f"λ={damping:.1E}")
                break

        return xy, rad2deg(yaw)


def compute_residuals(xy, yaw, feats_map, feats_q, grid_uv_cam, interp, grads=False):
    grid_uv_world = xy + grid_uv_cam @ rotmat2d(yaw).T
    feats_map_view, valid_view, J_f_p2d = interp(
        feats_map, grid_uv_world, return_gradients=grads
    )
    res = feats_map_view - feats_q
    return res, valid_view, J_f_p2d


def build_system(res, J, weights):
    grad = torch.einsum("...ndi,...nd->...ni", J, res)  # ... x N x 3
    grad = weights[..., None] * grad
    grad = grad.sum(-2)  # ... x 3

    Hess = torch.einsum("...ijk,...ijl->...ikl", J, J)  # ... x N x 3 x 3
    Hess = weights[..., None, None] * Hess
    Hess = Hess.sum(-3)  # ... x 3 x 3

    return grad, Hess


def optimizer_step(g, H, lambda_=0, mute=False, mask=None, eps=1e-6):
    """One optimization step with Gauss-Newton or Levenberg-Marquardt.
    Args:
        g: batched gradient tensor of size (..., N).
        H: batched hessian tensor of size (..., N, N).
        lambda_: damping factor for LM (use GN if lambda_=0).
        mask: denotes valid elements of the batch (optional).
    Adapted from PixLoc, Paul-Edouard Sarlin, ETH Zurich
    https://github.com/cvg/pixloc
    Released under the Apache License 2.0
    """
    if lambda_ == 0:
        diag = torch.zeros_like(g)
    else:
        diag = H.diagonal(dim1=-2, dim2=-1) * lambda_
    H = H + diag.clamp(min=eps).diag_embed()

    if mask is not None:
        # make sure that masked elements are not singular
        H = torch.where(mask[..., None, None], H, torch.eye(H.shape[-1]).to(H))
        # set g to 0 to delta is 0 for masked elements
        g = g.masked_fill(~mask[..., None], 0.0)

    H_, g_ = H.cpu(), g.cpu()
    try:
        U = torch.linalg.cholesky(H_)
    except RuntimeError as e:
        if "singular U" in str(e):
            if not mute:
                logger.debug("Cholesky decomposition failed, fallback to LU.")
            delta = -torch.solve(g_[..., None], H_)[0][..., 0]
        else:
            raise
    else:
        delta = -torch.cholesky_solve(g_[..., None], U)[..., 0]

    return delta.to(H.device)


def safe_solve_with_mask(
    B: torch.Tensor, A: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Helper function, which avoids crashing because of singular matrix input and outputs the
    mask of valid solution. Copied from Kornia
    at https://github.com/kornia/kornia/blob/902fc0eb7357d7792aa122a131f0c624d992046c/kornia/utils/helpers.py#L89
    """
    # Based on https://github.com/pytorch/pytorch/issues/31546#issuecomment-694135622
    if not isinstance(B, torch.Tensor):
        raise AssertionError(f"B must be torch.Tensor. Got: {type(B)}.")
    dtype: torch.dtype = B.dtype
    if dtype not in (torch.float32, torch.float64):
        dtype = torch.float32
    A_LU, pivots, info = torch.lu(A.to(dtype), get_infos=True)
    valid_mask: torch.Tensor = info == 0
    X = torch.lu_solve(B.to(dtype), A_LU, pivots)
    return X.to(B.dtype), A_LU.to(A.dtype), valid_mask


def subpixel_refinement(logprob_uvt, uv, yaw, max_delta=0.5):
    """
    Inspired by kornia.geometry.subpix.conv_quad_interp3d
    at https://github.com/kornia/kornia/blob/902fc0eb7357d7792aa122a131f0c624d992046c/kornia/geometry/subpix/spatial_soft_argmax.py#L563
    """
    assert len(logprob_uvt.shape) == 3
    ij = uv.flip(-1).round().long()
    k = (yaw / 360 * logprob_uvt.shape[-1]).round().long()

    # scores = torch.log(1 - logprob_uvt[None, None].exp())
    scores = logprob_uvt[None, None]
    scores = torch.nn.functional.pad(scores, [1, 2, 0, 0, 0, 0], mode="circular")
    scores = scores[:, :, ij[0] - 1 : ij[0] + 2, ij[1] - 1 : ij[1] + 2, k : k + 3]
    assert scores.shape == (1, 1, 3, 3, 3), (scores.shape, logprob_uvt.shape, ij, k)

    b: torch.Tensor = spatial_gradient3d(scores, order=1, mode="diff")
    b = b[0, 0, :, 1, 1, 1].view(1, 3, 1)
    A: torch.Tensor = spatial_gradient3d(scores, order=2, mode="diff")
    A = A[0, 0, :, 1, 1, 1][None]
    dxx = A[..., 0]
    dyy = A[..., 1]
    dss = A[..., 2]
    dxy = 0.25 * A[..., 3]  # normalization to match OpenCV implementation
    dys = 0.25 * A[..., 4]  # normalization to match OpenCV implementation
    dxs = 0.25 * A[..., 5]  # normalization to match OpenCV implementation

    Hes = torch.stack([dxx, dxy, dxs, dxy, dyy, dys, dxs, dys, dss], dim=-1).view(
        -1, 3, 3
    )
    xyt_solved, _, solved_correctly = safe_solve_with_mask(b, Hes)
    assert solved_correctly.all()
    delta = -xyt_solved.squeeze(0).squeeze(-1)
    diverged = delta.abs().max(-1, keepdim=True).values > max_delta
    delta = delta.masked_fill(diverged.expand_as(delta), 0)
    d_uv = delta[:2]
    d_yaw = delta[-1] * 360 / logprob_uvt.shape[-1]
    return uv + d_uv, yaw + d_yaw
