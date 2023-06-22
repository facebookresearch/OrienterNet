# Copyright (c) Meta Platforms, Inc. and affiliates.

import copy

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch


def likelihood_overlay(
    prob, map_viz=None, p_rgb=0.2, p_alpha=1 / 15, thresh=None, cmap="jet"
):
    prob = prob / prob.max()
    cmap = plt.get_cmap(cmap)
    rgb = cmap(prob**p_rgb)
    alpha = prob[..., None] ** p_alpha
    if thresh is not None:
        alpha[prob <= thresh] = 0
    if map_viz is not None:
        faded = map_viz + (1 - map_viz) * 0.5
        rgb = rgb[..., :3] * alpha + faded * (1 - alpha)
        rgb = np.clip(rgb, 0, 1)
    else:
        rgb[..., -1] = alpha.squeeze(-1)
    return rgb


def heatmap2rgb(scores, mask=None, clip_min=0.05, alpha=0.8, cmap="jet"):
    min_, max_ = np.quantile(scores, [clip_min, 1])
    scores = scores.clip(min=min_)
    rgb = plt.get_cmap(cmap)((scores - min_) / (max_ - min_))
    if mask is not None:
        if alpha == 0:
            rgb[mask] = np.nan
        else:
            rgb[..., -1] = 1 - (1 - 1.0 * mask) * (1 - alpha)
    return rgb


def plot_pose(axs, xy, yaw=None, s=1 / 35, c="r", a=1, w=0.015, dot=True, zorder=10):
    if yaw is not None:
        yaw = np.deg2rad(yaw)
        uv = np.array([np.sin(yaw), -np.cos(yaw)])
    xy = np.array(xy) + 0.5
    if not isinstance(axs, list):
        axs = [axs]
    for ax in axs:
        if isinstance(ax, int):
            ax = plt.gcf().axes[ax]
        if dot:
            ax.scatter(*xy, c=c, s=70, zorder=zorder, linewidths=0, alpha=a)
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


def plot_dense_rotations(
    ax, prob, thresh=0.01, skip=10, s=1 / 15, k=3, c="k", w=None, **kwargs
):
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
    plot_pose(
        ax,
        indices[::-1],
        yaws[indices],
        s=s,
        c=c,
        dot=False,
        zorder=0.1,
        w=w,
        **kwargs,
    )


def copy_image(im, ax):
    prop = im.properties()
    prop.pop("children")
    prop.pop("size")
    prop.pop("tightbbox")
    prop.pop("transformed_clip_path_and_affine")
    prop.pop("window_extent")
    prop.pop("figure")
    prop.pop("transform")
    return ax.imshow(im.get_array(), **prop)


def add_circle_inset(
    ax,
    center,
    corner=None,
    radius_px=10,
    inset_size=0.4,
    inset_offset=0.005,
    color="red",
):
    data_t_axes = ax.transAxes + ax.transData.inverted()
    if corner is None:
        center_axes = np.array(data_t_axes.inverted().transform(center))
        corner = 1 - np.round(center_axes).astype(int)
    corner = np.array(corner)
    bottom_left = corner * (1 - inset_size - inset_offset) + (1 - corner) * inset_offset
    axins = ax.inset_axes([*bottom_left, inset_size, inset_size])
    if ax.yaxis_inverted():
        axins.invert_yaxis()
    axins.set_axis_off()

    c = mpl.patches.Circle(center, radius_px, fill=False, color=color)
    ax.add_patch(copy.deepcopy(c))
    axins.add_patch(c)

    radius_inset = radius_px + 1
    axins.set_xlim([center[0] - radius_inset, center[0] + radius_inset])
    ylim = center[1] - radius_inset, center[1] + radius_inset
    if axins.yaxis_inverted():
        ylim = ylim[::-1]
    axins.set_ylim(ylim)

    for im in ax.images:
        im2 = copy_image(im, axins)
        im2.set_clip_path(c)
    return axins


def plot_bev(bev, uv, yaw, ax=None, zorder=10, **kwargs):
    if ax is None:
        ax = plt.gca()
    h, w = bev.shape[:2]
    tfm = mpl.transforms.Affine2D().translate(-w / 2, -h)
    tfm = tfm.rotate_deg(yaw).translate(*uv + 0.5)
    tfm += plt.gca().transData
    ax.imshow(bev, transform=tfm, zorder=zorder, **kwargs)
    ax.plot(
        [0, w - 1, w / 2, 0],
        [0, 0, h - 0.5, 0],
        transform=tfm,
        c="k",
        lw=1,
        zorder=zorder + 1,
    )
