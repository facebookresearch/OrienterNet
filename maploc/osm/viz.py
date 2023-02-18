# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from ..utils.geometry import to_homogeneous
from ..utils.viz_2d import add_text, plot_images, plot_keypoints
from ..utils.wrappers import Camera, Pose
from .parser import (
    filter_area,
    filter_node,
    filter_way,
    Groups,
    parse_area,
    parse_node,
    parse_way,
)


class MapPlotter:
    def __init__(self, proj):
        self.proj = proj

    def plot_tile(self, nodes, ways, bbox, T_w2c, dpi=150):
        self.bbox = bbox

        fig = plt.figure(dpi=dpi)
        plt.scatter(*bbox.center, c="r", s=5)
        vec = T_w2c.inv() @ (np.array([[-1, 0, 1], [1, 0, 1]]) * 2)
        plt.plot(*np.stack([vec[0, :2], bbox.center, vec[1, :2]], 1), "r", linewidth=1)

        ax = plt.gca()
        ax.set_xlim([bbox.min_[0], bbox.max_[0]])
        ax.set_ylim([bbox.min_[1], bbox.max_[1]])
        ax.get_yaxis().set_ticks([])
        ax.get_xaxis().set_ticks([])
        for sp in ax.spines.values():
            sp.set_color("k")
        ax.set_aspect("equal")
        ax.autoscale(enable=False)
        fig.tight_layout(pad=0)

        for node in filter(filter_node, nodes):
            self.plot_node(node)

        self.area_colors = {}
        for area in filter(filter_area, ways):
            self.plot_area(area)

        self.way_colors = {}
        for way in filter(filter_way, ways):
            self.plot_way(way)

        return fig

    def plot_node(self, node):
        label = parse_node(node.tags)
        if label is None:
            return
        xy = node.xy

        if label == "natural:tree":
            color = "g"
            size = 10
        else:
            color = "b"
            size = 5
            plt.text(
                *xy, label.split(":")[-1], fontsize=5, c="k", ha="center", clip_on=True
            )
        plt.scatter(*xy, c=color, s=size)

    def plot_area(self, area):
        label = parse_area(area.tags)
        if label is None:
            return
        xy = np.stack([n.xy for n in area.nodes])

        plt.fill(*xy.T, alpha=0.3, zorder=-1)
        plt.scatter(*xy.T, c="k", s=2)
        inbox = self.bbox.contains(xy)
        if np.count_nonzero(inbox):
            plt.text(
                *xy[inbox].mean(0).T,
                label,
                fontsize=5,
                c="k",
                ha="center",
                va="center",
                clip_on=True,
            )

        self.area_colors[area.id_] = plt.gca().patches[-1]._facecolor[:3]

    def plot_way(self, way):
        label = parse_way(way.tags)
        if label is None:
            return
        xy = np.stack([n.xy for n in way.nodes])

        if label.startswith("highway"):
            if label.split(":")[-1] in [
                "path",
                "footway",
                "steps",
                "sidewalk",
                "crossing",
            ]:
                color = "red"
            else:
                color = "lime"
        else:
            color = "fuchsia"
        plt.plot(*xy.T, alpha=1.0, zorder=-1, linewidth=2, c=color)
        if not label.startswith("highway"):
            inbox = self.bbox.contains(xy)
            if np.count_nonzero(inbox):
                plt.text(
                    *xy[inbox].mean(0).T,
                    label,
                    fontsize=5,
                    c="k",
                    ha="center",
                    va="center",
                    clip_on=True,
                )

        self.way_colors[way.id_] = mpl.colors.to_rgb(color)

    def draw_raster(self, nodes, ways, canvas):
        raster = []
        colors = []

        for area in filter(filter_area, ways):
            label = parse_area(area.tags)
            if label is None:
                continue
            xy = np.stack([n.xy for n in area.nodes])
            canvas.draw_polygon(xy)
            raster.append(canvas.raster)
            canvas.clear()
            colors.append(self.area_colors[area.id_])

        for way in filter(filter_way, ways):
            label = parse_way(way.tags)
            if label is None:
                continue
            xy = np.stack([n.xy for n in way.nodes])
            canvas.draw_line(xy, 3)
            raster.append(canvas.raster)
            canvas.clear()
            colors.append(self.way_colors[way.id_])

        raster = np.stack(raster, 0)
        asort = np.argsort(raster.sum((1, 2)))
        raster = np.argmax(
            np.concatenate([np.zeros_like(raster[:1]), raster[asort]]), 0
        )
        raster = np.concatenate([[[0, 0, 0]], np.array(colors)[asort]])[raster]
        return raster


def plot_overlay(
    T_w2c: Pose,
    camera: Camera,
    z_offset: float,
    image: np.ndarray,
    proj,
    bbox,
    canvas,
    raster,
    data,
    text,
):

    z_ground = T_w2c.inv().t[-1] - z_offset
    h, w = image.shape[:2]
    grid = np.mgrid[:h, :w][::-1]
    p2d = grid.reshape(2, -1).T
    xyz = to_homogeneous(camera.denormalize(p2d)) @ T_w2c.R
    scale = -z_offset / xyz[:, -1]
    xyz = xyz * scale[:, None] + T_w2c.inv().t
    valid = scale > 0
    valid &= bbox.contains(xyz[:, :2])

    uv = canvas.to_uv(xyz[valid, :2])
    colors = raster[tuple(np.round(uv).astype(int).T[::-1])]
    map_proj = np.zeros((len(p2d), 3))
    map_proj[np.where(valid)] = colors
    map_proj = map_proj.reshape(image.shape)
    mask = ~np.all(map_proj == [0, 0, 0], -1)

    overlay = np.where(
        mask[:, :, None], image / 255 * 0.3 + map_proj * 0.7, image / 255
    )
    plot_images([overlay], dpi=150)
    plt.gca().autoscale(enable=False)
    add_text(0, text, fs=5, pos=(0.01, 0.01), va="bottom")

    for node in filter(filter_node, data.nodes.values()):
        label = parse_node(node.tags)
        if label is None:
            continue
        xy = proj.project(node.geo)
        if xy is None:
            continue
        if bbox.contains(xy):
            p2d, valid = camera.project(T_w2c @ np.r_[xy, z_ground])
            if valid:
                plot_keypoints([p2d[None]], ps=20, colors="b")
                plt.text(*p2d, label.split(":")[-1], c="k", fontsize=10, clip_on=True)

    return plt.gcf()


def plot_locations(locations, bbox):
    plt.scatter(*(locations - bbox.center).T, s=0.1)
    plt.gca().add_patch(
        mpl.patches.Rectangle(
            bbox.min_ - bbox.center, *(bbox.max_ - bbox.min_), lw=1, ec="r", fc="none"
        )
    )
    plt.gca().set_aspect("equal")


class Colormap:
    def __init__(self):
        self.colors = {
            "building": (84, 155, 255),
            "parking": (255, 229, 145),
            "playground": (150, 133, 125),
            "grass": (188, 255, 143),
            "park": (0, 158, 16),
            "forest": (0, 92, 9),
            "water": (184, 213, 255),
            "fence": (238, 0, 255),
            "wall": (0, 0, 0),
            "hedge": (107, 68, 48),
            "kerb": (255, 234, 0),
            "building_outline": (0, 0, 255),
            "cycleway": (0, 251, 255),
            "path": (8, 237, 0),
            "road": (255, 0, 0),
            "tree_row": (0, 92, 9),
            "busway": (255, 128, 0),
            "void": [int(255 * 0.9)] * 3,
        }
        self.colors_areas = np.stack([self.colors[k] for k in ["void"] + Groups.areas])
        self.colors_ways = np.stack([self.colors[k] for k in ["void"] + Groups.ways])

    def apply(self, rasters):
        return (
            np.where(
                rasters[1, ..., None] > 0,
                self.colors_ways[rasters[1]],
                self.colors_areas[rasters[0]],
            )
            / 255.0
        )

    def add_colorbar(self):
        ax2 = plt.gcf().add_axes([1, 0.1, 0.02, 0.8])
        color_list = np.r_[self.colors_areas[1:], self.colors_ways[1:]] / 255.0
        cmap = mpl.colors.LinearSegmentedColormap.from_list(
            "Custom cmap", color_list, len(color_list)
        )
        bounds = np.arange(len(color_list) + 1)
        cb = mpl.colorbar.ColorbarBase(
            ax2,
            cmap=cmap,
            orientation="vertical",
            ticks=bounds[:-1] + 0.5,
            boundaries=bounds,
            format="%1i",
        )
        cb.set_ticklabels(Groups.areas + Groups.ways)
        ax2.tick_params(labelsize=15)


def plot_nodes(idx, raster):
    ax = plt.gcf().axes[idx]
    ax.autoscale(enable=False)
    nodes_xy = np.stack(np.where(raster > 0)[::-1], -1)
    nodes_val = raster[tuple(nodes_xy.T[::-1])] - 1
    ax.scatter(*nodes_xy.T, c="k", s=15)
    for xy, val in zip(nodes_xy, nodes_val):
        group = Groups.nodes[val]
        add_text(
            idx,
            group,
            xy + 2,
            lcolor=None,
            fs=8,
            color="k",
            normalized=False,
            ha="center",
        )
