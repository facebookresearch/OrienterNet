# Copyright (c) Meta Platforms, Inc. and affiliates.

# Adapted from Hierarchical-Localization, Paul-Edouard Sarlin, ETH Zurich
# https://github.com/cvg/Hierarchical-Localization/blob/master/hloc/utils/viz.py
# Released under the Apache License 2.0

import matplotlib
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np


def plot_images(imgs, titles=None, cmaps="gray", dpi=100, pad=0.5, adaptive=True):
    """Plot a set of images horizontally.
    Args:
        imgs: a list of NumPy or PyTorch images, RGB (H, W, 3) or mono (H, W).
        titles: a list of strings, as titles for each image.
        cmaps: colormaps for monochrome images.
        adaptive: whether the figure size should fit the image aspect ratios.
    """
    n = len(imgs)
    if not isinstance(cmaps, (list, tuple)):
        cmaps = [cmaps] * n

    if adaptive:
        ratios = [i.shape[1] / i.shape[0] for i in imgs]  # W / H
    else:
        ratios = [4 / 3] * n
    figsize = [sum(ratios) * 4.5, 4.5]
    fig, ax = plt.subplots(
        1, n, figsize=figsize, dpi=dpi, gridspec_kw={"width_ratios": ratios}
    )
    if n == 1:
        ax = [ax]
    for i in range(n):
        ax[i].imshow(imgs[i], cmap=plt.get_cmap(cmaps[i]))
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        ax[i].set_axis_off()
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
        if titles:
            ax[i].set_title(titles[i])
    fig.tight_layout(pad=pad)


def plot_keypoints(kpts, colors="lime", ps=4):
    """Plot keypoints for existing images.
    Args:
        kpts: list of ndarrays of size (N, 2).
        colors: string, or list of list of tuples (one for each keypoints).
        ps: size of the keypoints as float.
    """
    if not isinstance(colors, list):
        colors = [colors] * len(kpts)
    axes = plt.gcf().axes
    for a, k, c in zip(axes, kpts, colors):
        a.scatter(k[:, 0], k[:, 1], c=c, s=ps, linewidths=0)


def plot_matches(kpts0, kpts1, color=None, lw=1.5, ps=4, indices=(0, 1), a=1.0):
    """Plot matches for a pair of existing images.
    Args:
        kpts0, kpts1: corresponding keypoints of size (N, 2).
        color: color of each match, string or RGB tuple. Random if not given.
        lw: width of the lines.
        ps: size of the end points (no endpoint if ps=0)
        indices: indices of the images to draw the matches on.
        a: alpha opacity of the match lines.
    """
    fig = plt.gcf()
    ax = fig.axes
    assert len(ax) > max(indices)
    ax0, ax1 = ax[indices[0]], ax[indices[1]]
    fig.canvas.draw()

    assert len(kpts0) == len(kpts1)
    if color is None:
        color = matplotlib.cm.hsv(np.random.rand(len(kpts0))).tolist()
    elif len(color) > 0 and not isinstance(color[0], (tuple, list)):
        color = [color] * len(kpts0)

    if lw > 0:
        # transform the points into the figure coordinate system
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(ax0.transData.transform(kpts0))
        fkpts1 = transFigure.transform(ax1.transData.transform(kpts1))
        fig.lines += [
            matplotlib.lines.Line2D(
                (fkpts0[i, 0], fkpts1[i, 0]),
                (fkpts0[i, 1], fkpts1[i, 1]),
                zorder=1,
                transform=fig.transFigure,
                c=color[i],
                linewidth=lw,
                alpha=a,
            )
            for i in range(len(kpts0))
        ]

    # freeze the axes to prevent the transform to change
    ax0.autoscale(enable=False)
    ax1.autoscale(enable=False)

    if ps > 0:
        ax0.scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
        ax1.scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)


def add_text(
    idx,
    text,
    pos=(0.01, 0.99),
    fs=15,
    color="w",
    lcolor="k",
    lwidth=2,
    ha="left",
    va="top",
    normalized=True,
    zorder=3,
):
    ax = plt.gcf().axes[idx]
    tfm = ax.transAxes if normalized else ax.transData
    t = ax.text(
        *pos,
        text,
        fontsize=fs,
        ha=ha,
        va=va,
        color=color,
        transform=tfm,
        clip_on=True,
        zorder=zorder,
    )
    if lcolor is not None:
        t.set_path_effects(
            [
                path_effects.Stroke(linewidth=lwidth, foreground=lcolor),
                path_effects.Normal(),
            ]
        )


def save_plot(path, **kw):
    """Save the current figure without any white margin."""
    plt.savefig(path, bbox_inches="tight", pad_inches=0, **kw)


def features_to_RGB(*Fs, masks=None, skip=1):
    """Project a list of d-dimensional feature maps to RGB colors using PCA."""
    from sklearn.decomposition import PCA

    def normalize(x):
        return x / np.linalg.norm(x, axis=-1, keepdims=True)

    if masks is not None:
        assert len(Fs) == len(masks)

    flatten = []
    for i, F in enumerate(Fs):
        c, h, w = F.shape
        F = np.rollaxis(F, 0, 3)
        F_flat = F.reshape(-1, c)
        if masks is not None and masks[i] is not None:
            mask = masks[i]
            assert mask.shape == F.shape[:2]
            F_flat = F_flat[mask.reshape(-1)]
        flatten.append(F_flat)
    flatten = np.concatenate(flatten, axis=0)
    flatten = normalize(flatten)

    pca = PCA(n_components=3)
    if skip > 1:
        pca.fit(flatten[::skip])
        flatten = pca.transform(flatten)
    else:
        flatten = pca.fit_transform(flatten)
    flatten = (normalize(flatten) + 1) / 2

    Fs_rgb = []
    for i, F in enumerate(Fs):
        h, w = F.shape[-2:]
        if masks is None or masks[i] is None:
            F_rgb, flatten = np.split(flatten, [h * w], axis=0)
            F_rgb = F_rgb.reshape((h, w, 3))
        else:
            F_rgb = np.zeros((h, w, 3))
            indices = np.where(masks[i])
            F_rgb[indices], flatten = np.split(flatten, [len(indices[0])], axis=0)
            F_rgb = np.concatenate([F_rgb, masks[i][..., None]], axis=-1)
        Fs_rgb.append(F_rgb)
    assert flatten.shape[0] == 0, flatten.shape
    return Fs_rgb
