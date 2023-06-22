# Copyright (c) Meta Platforms, Inc. and affiliates.

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import PIL.Image

from ..utils.viz_2d import add_text
from .parser import Groups


class GeoPlotter:
    def __init__(self, zoom=12, **kwargs):
        self.fig = go.Figure()
        self.fig.update_layout(
            mapbox_style="open-street-map",
            autosize=True,
            mapbox_zoom=zoom,
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            showlegend=True,
            **kwargs,
        )

    def points(self, latlons, color, text=None, name=None, size=5, **kwargs):
        latlons = np.asarray(latlons)
        self.fig.add_trace(
            go.Scattermapbox(
                lat=latlons[..., 0],
                lon=latlons[..., 1],
                mode="markers",
                text=text,
                marker_color=color,
                marker_size=size,
                name=name,
                **kwargs,
            )
        )
        center = latlons.reshape(-1, 2).mean(0)
        self.fig.update_layout(
            mapbox_center=dict(zip(("lat", "lon"), center)),
        )

    def bbox(self, bbox, color, name=None, **kwargs):
        corners = np.stack(
            [bbox.min_, bbox.left_top, bbox.max_, bbox.right_bottom, bbox.min_]
        )
        self.fig.add_trace(
            go.Scattermapbox(
                lat=corners[:, 0],
                lon=corners[:, 1],
                mode="lines",
                marker_color=color,
                name=name,
                **kwargs,
            )
        )
        self.fig.update_layout(
            mapbox_center=dict(zip(("lat", "lon"), bbox.center)),
        )

    def raster(self, raster, bbox, below="traces", **kwargs):
        if not np.issubdtype(raster.dtype, np.integer):
            raster = (raster * 255).astype(np.uint8)
        raster = PIL.Image.fromarray(raster)
        corners = np.stack(
            [
                bbox.min_,
                bbox.left_top,
                bbox.max_,
                bbox.right_bottom,
            ]
        )[::-1, ::-1]
        layers = [*self.fig.layout.mapbox.layers]
        layers.append(
            dict(
                sourcetype="image",
                source=raster,
                coordinates=corners,
                below=below,
                **kwargs,
            )
        )
        self.fig.layout.mapbox.layers = layers


map_colors = {
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


class Colormap:
    colors_areas = np.stack([map_colors[k] for k in ["void"] + Groups.areas])
    colors_ways = np.stack([map_colors[k] for k in ["void"] + Groups.ways])

    @classmethod
    def apply(cls, rasters):
        return (
            np.where(
                rasters[1, ..., None] > 0,
                cls.colors_ways[rasters[1]],
                cls.colors_areas[rasters[0]],
            )
            / 255.0
        )

    @classmethod
    def add_colorbar(cls):
        ax2 = plt.gcf().add_axes([1, 0.1, 0.02, 0.8])
        color_list = np.r_[cls.colors_areas[1:], cls.colors_ways[1:]] / 255.0
        cmap = mpl.colors.ListedColormap(color_list[::-1])
        ticks = np.linspace(0, 1, len(color_list), endpoint=False)
        ticks += 1 / len(color_list) / 2
        cb = mpl.colorbar.ColorbarBase(
            ax2,
            cmap=cmap,
            orientation="vertical",
            ticks=ticks,
        )
        cb.set_ticklabels((Groups.areas + Groups.ways)[::-1])
        ax2.tick_params(labelsize=15)


def plot_nodes(idx, raster, fontsize=8, size=15):
    ax = plt.gcf().axes[idx]
    ax.autoscale(enable=False)
    nodes_xy = np.stack(np.where(raster > 0)[::-1], -1)
    nodes_val = raster[tuple(nodes_xy.T[::-1])] - 1
    ax.scatter(*nodes_xy.T, c="k", s=size)
    for xy, val in zip(nodes_xy, nodes_val):
        group = Groups.nodes[val]
        add_text(
            idx,
            group,
            xy + 2,
            lcolor=None,
            fs=fontsize,
            color="k",
            normalized=False,
            ha="center",
        )
