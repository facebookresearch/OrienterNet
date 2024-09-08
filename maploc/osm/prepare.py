import logging
from enum import Enum, auto
from pathlib import Path

from ..utils.geo import BoundaryBox, Projection
from ..utils.io import DATA_URL, download_file
from .download import convert_osm_file, get_geofabrik_url
from .tiling import TileManager

logger = logging.getLogger(__name__)


class OSMDataSource(Enum):
    # Pre-computed map tiles.
    PRECOMPUTED = auto()

    # Re-compute the map tiles from cached OSM data.
    CACHED = auto()

    # Fetch the latest OSM data and re-compute the map tiles from them.
    LATEST = auto()


def download_and_prepare_osm(
    source: OSMDataSource,
    tiles_name: str,
    tiles_path: Path,
    bbox: BoundaryBox,
    projection: Projection,
    osm_path: Path,
    **kwargs,
) -> TileManager:
    if source == OSMDataSource.PRECOMPUTED:
        if not tiles_path.exists():
            logger.info("Downloading pre-computed map tiles.")
            download_file(DATA_URL + f"/tiles/{tiles_name}.pkl", tiles_path)
        tile_manager = TileManager.load(tiles_path)
        assert tile_manager.ppm == kwargs["ppm"]
        assert tile_manager.bbox.contains(bbox)
    else:
        logger.info("Creating the map tiles.")
        if source == OSMDataSource.CACHED:
            if not osm_path.exists():
                logger.info("Downloading cached OSM data.")
                download_file(DATA_URL + f"/osm/{osm_path.name}", osm_path)
            if not osm_path.exists():
                raise FileNotFoundError(f"Cannot find OSM data file {osm_path}.")
        elif source == OSMDataSource.LATEST:
            logger.info("Downloading the latest OSM data.")
            bbox_osm = projection.unproject(bbox + 2_000)  # 2 km
            url = get_geofabrik_url(bbox_osm)
            tmp_path = osm_path.parent / Path(url).name
            download_file(url, tmp_path)
            convert_osm_file(bbox_osm, tmp_path, osm_path)
            tmp_path.unlink()
        else:
            raise NotImplementedError("Unknown source: {osm_source}.")
        tile_manager = TileManager.from_bbox(
            projection,
            bbox,
            path=osm_path,
            **kwargs,
        )
        tile_manager.save(tiles_path)
    return tile_manager
