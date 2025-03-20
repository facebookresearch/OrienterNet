from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from maploc.osm.tiling import TileManager
from maploc.osm.viz import Colormap, GeoPlotter, plot_nodes
from maploc.utils.exif import EXIF
from maploc.utils.geo import BoundaryBox, Projection
from maploc.utils.viz_2d import plot_images


# Custom Overrides to avoid pytorch
def parse_location_prior(
    exif: EXIF,
    prior_latlon: Optional[Tuple[float, float]] = None,
    prior_address: Optional[str] = None,
) -> np.ndarray:
    latlon = None
    if latlon is None:
        geo = exif.extract_geo()
        if geo:
            alt = geo.get("altitude", 0)  # read if available
            latlon = (geo["latitude"], geo["longitude"], alt)
            # logger.info("Using prior location from EXIF.")
        else:
            raise ValueError(
                "No location prior given or found in the image EXIF metadata: "
                "maybe provide the name of a street, building or neighborhood?"
            )
    return np.array(latlon)


def read_input_image(
    image_path: str,
    prior_latlon: Optional[Tuple[float, float]] = None,
    prior_address: Optional[str] = None,
    focal_length: Optional[float] = None,
    tile_size_meters: int = 64,
) -> Tuple[np.ndarray, Tuple[str, str], Projection, BoundaryBox]:
    image = read_image(image_path)
    with open(image_path, "rb") as fid:
        exif = EXIF(fid, lambda: image.shape[:2])

        # gravity, camera = self.calibrator.run(image, focal_length, exif)
        # logger.info("Using (roll, pitch) %s.", gravity)
        print(exif)

        latlon = parse_location_prior(exif, prior_latlon, prior_address)
        proj = Projection(*latlon)
        center = proj.project(latlon)
        bbox = BoundaryBox(center, center) + tile_size_meters
        return image, None, proj, bbox


def read_image(path, grayscale=False):
    if grayscale:
        mode = cv2.IMREAD_GRAYSCALE
    else:
        mode = cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise ValueError(f"Cannot read image {path}.")
    if not grayscale and len(image.shape) == 3:
        image = np.ascontiguousarray(image[:, :, ::-1])  # BGR to RGB
    return image

    # no EXIF data: provide a coarse location prior as address


image_path = "assets/query_zurich_1.JPG"
prior_address = "ETH CAB Zurich"

# Try out these other queries!
# image_path = "assets/query_vancouver_1.JPG"
# prior_address = "Vancouver Waterfront Station"

# image_path = "assets/query_vancouver_2.JPG"
image_path = "assets/query_vancouver_3.JPG"
# prior_address = None # here we load the location prior from the exif

image, gravity, proj, bbox = read_input_image(
    image_path,
    tile_size_meters=128,  # try 64, 256, etc.
)

# Show the query area in an interactive map
# plot = GeoPlotter(zoom=16)
# plot.points(proj.latlonalt[:2], "red", name="location prior", size=10)
# plot.bbox(proj.unproject(bbox), "blue", name="map tile")
# plot.fig.show()


# Query OpenStreetMap for this area
# tiler = TileManager.from_bbox(proj, bbox + 10, 2, Path("tscache.json"))
# canvas = tiler.query(bbox)
# map_viz = Colormap.apply(canvas.raster)
# cv2.imwrite("t-osm.png", map_viz)

# print(
#     f"""
#     Lines: {len(tiler.map_data.lines)}
#     Areas: {len(tiler.map_data.areas)}
#     Points: {len(tiler.map_data.nodes)}"""
# )

print("starting...")

tiler2 = TileManager.from_bbox(proj, bbox + 10, 2, source='vector-tiles')

print("done")
canvas2 = tiler2.query(bbox)

print(
    f"""
    Lines: {len(tiler2.map_data.lines)}
    Areas: {len(tiler2.map_data.areas)}
    Points: {len(tiler2.map_data.nodes)}"""
)
map_viz2 = Colormap.apply(canvas2.raster)
cv2.imwrite("t-vector-tiles.png", map_viz2)

# plot_images([image, map_viz], titles=["input image", "OpenStreetMap raster"])
