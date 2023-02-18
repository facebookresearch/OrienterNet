# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from typing import Dict, List, Optional, Tuple

import folium


def plot_trajectories_on_map(
    coordinates_lists: Dict[str, List[Tuple[float, float]]],
    aoi: Optional[List[Tuple[float, float]]] = None,
    tiles_type="OpenStreetMap",
    polyline: bool = False,
):
    color_pallete = [
        "red",
        "blue",
        "green",
        "purple",
        "orange",
        "darkred",
        "lightred",
        "darkblue",
        "darkgreen",
        "cadetblue",
        "darkpurple",
        "pink",
        "lightblue",
        "lightgreen",
        "gray",
        "black",
        "lightgray",
    ]

    lat0, lon0 = next(iter(coordinates_lists.values()))[0]
    m = folium.Map(
        location=[lat0, lon0],
        control_scale=True,
        zoom_start=18,
        max_zoom=19,
        tiles=tiles_type,
    )
    for i, (name, coor_list) in enumerate(coordinates_lists.items()):
        color = color_pallete[i % len(color_pallete)]
        fg = folium.FeatureGroup(name)
        if polyline:
            folium.PolyLine(coor_list, color=color).add_to(fg)
        else:
            for lat, lon in coor_list:
                folium.Circle([lat, lon], radius=1, color=color).add_to(fg)
        fg.add_to(m)

    if aoi:
        aoi_json = {
            "type": "Feature",
            "id": 0,
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[lon, lat] for lat, lon in aoi]],
            },
        }
        folium.GeoJson(aoi_json).add_to(m)

    folium.LayerControl().add_to(m)
    folium.LatLngPopup().add_to(m)

    return m
