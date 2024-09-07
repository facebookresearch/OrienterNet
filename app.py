import csv
import sys

import gradio as gr
import matplotlib.pyplot as plt

from maploc.demo import Demo
from maploc.osm.tiling import TileManager
from maploc.osm.viz import Colormap, GeoPlotter, plot_nodes
from maploc.utils.viz_2d import features_to_RGB, plot_images
from maploc.utils.viz_localization import (
    add_circle_inset,
    likelihood_overlay,
    plot_dense_rotations,
)

# Avoid `_csv.Error: field larger than field limit` with large plots.
csv.field_size_limit(sys.maxsize)

# Fixes https://github.com/gradio-app/gradio/issues/9287
gr.utils.sanitize_value_for_csv = lambda v: v


def run(image, address, tile_size_meters, num_rotations):
    image_path = image.name
    demo = Demo(num_rotations=int(num_rotations))

    try:
        image, camera, gravity, proj, bbox = demo.read_input_image(
            image_path,
            prior_address=address or None,
            tile_size_meters=int(tile_size_meters),
        )
    except ValueError as e:
        raise gr.Error(str(e))

    tiler = TileManager.from_bbox(proj, bbox + 10, demo.config.data.pixel_per_meter)
    canvas = tiler.query(bbox)
    map_viz = Colormap.apply(canvas.raster)

    plot_images([image, map_viz], titles=["input image", "OpenStreetMap raster"], pad=2)
    plot_nodes(1, canvas.raster[2], fontsize=6, size=10)
    fig1 = plt.gcf()

    # Run the inference
    try:
        uv, yaw, prob, neural_map, image_rectified = demo.localize(
            image, camera, canvas, gravity=gravity
        )
    except RuntimeError as e:
        raise gr.Error(str(e))

    # Visualize the predictions
    overlay = likelihood_overlay(prob.numpy().max(-1), map_viz.mean(-1, keepdims=True))
    (neural_map_rgb,) = features_to_RGB(neural_map.numpy())
    plot_images([overlay, neural_map_rgb], titles=["heatmap", "neural map"], pad=2)
    ax = plt.gcf().axes[0]
    ax.scatter(*canvas.to_uv(bbox.center), s=5, c="red")
    plot_dense_rotations(ax, prob, w=0.005, s=1 / 25)
    add_circle_inset(ax, uv)
    fig2 = plt.gcf()

    # Plot as interactive figure
    latlon = proj.unproject(canvas.to_xy(uv))
    bbox_latlon = proj.unproject(canvas.bbox)
    plot = GeoPlotter(zoom=16.5)
    plot.raster(map_viz, bbox_latlon, opacity=0.5)
    plot.raster(likelihood_overlay(prob.numpy().max(-1)), proj.unproject(bbox))
    plot.points(proj.latlonalt[:2], "red", name="location prior", size=10)
    plot.points(latlon, "black", name="argmax", size=10, visible="legendonly")
    plot.bbox(bbox_latlon, "blue", name="map tile")

    coordinates = f"(latitude, longitude) = {tuple(map(float, latlon))}"
    coordinates += f"\nheading angle = {yaw:.2f}°"
    return fig1, fig2, plot.fig, coordinates


examples = [
    ["assets/query_zurich_1.JPG", "ETH CAB Zurich", 128, 256],
    ["assets/query_vancouver_1.JPG", "Vancouver Waterfront Station", 128, 256],
    ["assets/query_vancouver_2.JPG", None, 128, 256],
    ["assets/query_vancouver_3.JPG", None, 128, 256],
]

description = """
<h1 align="center">
  <ins>OrienterNet</ins>
  <br>
  Visual Localization in 2D Public Maps
  <br>
  with Neural Matching</h1>
<h3 align="center">
    <a href="https://psarlin.com/orienternet" target="_blank">Project Page</a> |
    <a href="https://arxiv.org/pdf/2304.02009.pdf" target="_blank">Paper</a> |
    <a href="https://github.com/facebookresearch/OrienterNet" target="_blank">Code</a> |
    <a href="https://youtu.be/wglW8jnupSs" target="_blank">Video</a>
</h3>
<p align="center">
OrienterNet finds the position and orientation of any image using OpenStreetMap.
Click on one of the provided examples or upload your own image!
</p>
"""

app = gr.Interface(
    fn=run,
    inputs=[
        gr.File(file_types=["image"]),
        gr.Textbox(
            label="Prior location (optional)",
            info="Required if the image metadata (EXIF) does not contain a GPS prior. "
            "Enter an address or a street or building name.",
        ),
        gr.Radio(
            [64, 128, 256, 512],
            value=128,
            label="Search radius (meters)",
            info="Depends on how coarse the prior location is.",
        ),
        gr.Radio(
            [64, 128, 256, 360],
            value=256,
            label="Number of rotations",
            info="Reduce to scale to larger areas.",
        ),
    ],
    outputs=[
        gr.Plot(label="Inputs"),
        gr.Plot(label="Outputs"),
        gr.Plot(label="Interactive map"),
        gr.Textbox(label="Predicted coordinates"),
    ],
    description=description,
    examples=examples,
    cache_examples=True,
)

app.launch(share=False)
