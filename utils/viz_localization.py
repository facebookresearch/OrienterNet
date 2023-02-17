# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch


def likelihood_overlay(prob, map_viz, power=1 / 5, thresh=0.01):
    prob = prob / prob.max()
    keep = prob > thresh
    faded = map_viz + (1 - map_viz) * 0.5
    alpha = prob[..., None] ** power
    overlay = np.where(
        keep[..., None], mpl.cm.jet(prob)[..., :3] * alpha + faded * (1 - alpha), faded
    )
    overlay = np.clip(overlay, 0, 1)
    return overlay


def plot_pose(idx, xy, yaw=None, s=0.01, c="r", a=1, w=None, dot=True, zorder=10):
    if yaw is not None:
        yaw = np.deg2rad(yaw)
        uv = np.array([np.sin(yaw), -np.cos(yaw)])
    xy = np.array(xy) + 0.5
    if not isinstance(idx, list):
        idx = [idx]
    for i in idx:
        ax = plt.gcf().axes[i]
        if dot:
            ax.scatter(*xy, c=c, s=70, zorder=zorder, linewidths=0)
        if yaw is not None:
            ax.quiver(
                *xy,
                *uv,
                scale=s,
                scale_units="xy",
                angles="xy",
                color=c,
                zorder=zorder,
                alpha=a,
                width=w,
            )


def plot_dense_rotations(i, prob, thresh=0.01, skip=10, s=1 / 25, k=3, c="k"):
    t = torch.argmax(prob, -1)
    yaws = t.numpy() / prob.shape[-1] * 360
    prob = prob.max(-1).values / prob.max()
    mask = prob > thresh
    masked = prob.masked_fill(~mask, 0)
    max_ = torch.nn.functional.max_pool2d(
        masked.float()[None, None], k, stride=1, padding=k // 2
    )
    mask = (max_[0, 0] == masked.float()) & mask
    indices = np.where(mask.numpy() > 0)
    plot_pose(i, indices[::-1], yaws[indices], s=s, c=c, a=0.8, dot=False, zorder=0.1)
