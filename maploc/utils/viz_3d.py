# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

# Adapted from PixLoc, Paul-Edouard Sarlin, ETH Zurich
# https://github.com/cvg/pixloc
# Released under the Apache License 2.0

"""
3D visualization primitives based on Plotly.
We might want to instead use a more powerful library like Open3D.
Plotly however supports animations, buttons and sliders.

1) Initialize a figure with `fig = init_figure()`
2) Plot points, cameras, lines, or create a slider animation.
3) Call `fig.show()` to render the figure.
"""

from typing import Optional

import numpy as np
import plotly.graph_objects as go

from .geometry import to_homogeneous


def init_figure(height=800):
    """Initialize a 3D figure."""
    fig = go.Figure()
    fig.update_layout(
        height=height,
        scene_camera=dict(eye=dict(x=0.0, y=-0.1, z=-2), up=dict(x=0, y=-1.0, z=0)),
        scene=dict(
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            aspectmode="data",
            dragmode="orbit",
        ),
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
    )  # noqa E741
    return fig


def plot_points(fig, pts, color="rgba(255, 0, 0, 1)", ps=2, lw=0.2):
    """Plot a set of 3D points."""
    x, y, z = pts.T
    tr = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        marker_size=ps,
        marker_color=color,
        marker_line_width=lw,
    )
    fig.add_trace(tr)


def plot_camera(
    fig: go.Figure,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    color: str = "rgb(0, 0, 255)",
    name: Optional[str] = None,
    legendgroup: Optional[str] = None,
    size: float = 1.0,
):
    """Plot a camera frustum from pose and intrinsic matrix."""
    W, H = K[0, 2] * 2, K[1, 2] * 2
    corners = np.array([[0, 0], [W, 0], [W, H], [0, H], [0, 0]])
    if size is not None:
        image_extent = max(size * W / 1024.0, size * H / 1024.0)
        world_extent = max(W, H) / (K[0, 0] + K[1, 1]) / 0.5
        scale = 0.5 * image_extent / world_extent
    else:
        scale = 1.0
    corners = to_homogeneous(corners) @ np.linalg.inv(K).T
    corners = (corners / 2 * scale) @ R.T + t
    corners = np.concatenate(([t], corners))
    indices = [1, 2, 3, 4, 1, 0, 2, 0, 3, 0, 4]
    x, y, z = np.stack([corners[i] for i in indices]).T

    pyramid = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        legendgroup=legendgroup,
        name=name,
        line=dict(color=color, width=1),
        showlegend=False,
    )
    fig.add_trace(pyramid)


def create_slider_animation(fig, traces):
    """Create a slider that animates a list of traces (e.g. 3D points)."""
    slider = {"steps": []}
    frames = []
    fig.add_trace(traces[0])
    idx = len(fig.data) - 1
    for i, tr in enumerate(traces):
        frames.append(go.Frame(name=str(i), traces=[idx], data=[tr]))
        step = {
            "args": [[str(i)], {"frame": {"redraw": True}, "mode": "immediate"}],
            "label": i,
            "method": "animate",
        }
        slider["steps"].append(step)
    fig.frames = tuple(frames)
    fig.layout.sliders = (slider,)
