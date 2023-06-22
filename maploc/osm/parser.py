# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
import re
from typing import List

from .reader import OSMData, OSMElement, OSMNode, OSMWay

IGNORE_TAGS = {"source", "phone", "entrance", "inscription", "note", "name"}


def parse_levels(string: str) -> List[float]:
    """Parse string representation of level sequence value."""
    try:
        cleaned = string.replace(",", ";").replace(" ", "")
        return list(map(float, cleaned.split(";")))
    except ValueError:
        logging.debug("Cannot parse level description from `%s`.", string)
        return []


def filter_level(elem: OSMElement):
    level = elem.tags.get("level")
    if level is not None:
        levels = parse_levels(level)
        # In the US, ground floor levels are sometimes marked as level=1
        # so let's be conservative and include it.
        if not (0 in levels or 1 in levels):
            return False
    layer = elem.tags.get("layer")
    if layer is not None:
        layer = parse_levels(layer)
        if len(layer) > 0 and max(layer) < 0:
            return False
    return (
        elem.tags.get("location") != "underground"
        and elem.tags.get("parking") != "underground"
    )


def filter_node(node: OSMNode):
    return len(node.tags.keys() - IGNORE_TAGS) > 0 and filter_level(node)


def is_area(way: OSMWay):
    if way.nodes[0] != way.nodes[-1]:
        return False
    if way.tags.get("area") == "no":
        return False
    filters = [
        "area",
        "building",
        "amenity",
        "indoor",
        "landuse",
        "landcover",
        "leisure",
        "public_transport",
        "shop",
    ]
    for f in filters:
        if f in way.tags and way.tags.get(f) != "no":
            return True
    if way.tags.get("natural") in {"wood", "grassland", "water"}:
        return True
    return False


def filter_area(way: OSMWay):
    return len(way.tags.keys() - IGNORE_TAGS) > 0 and is_area(way) and filter_level(way)


def filter_way(way: OSMWay):
    return not filter_area(way) and way.tags != {} and filter_level(way)


def parse_node(tags):
    keys = tags.keys()
    for key in [
        "amenity",
        "natural",
        "highway",
        "barrier",
        "shop",
        "tourism",
        "public_transport",
        "emergency",
        "man_made",
    ]:
        if key in keys:
            if "disused" in tags[key]:
                continue
            return f"{key}:{tags[key]}"
    return None


def parse_area(tags):
    if "building" in tags:
        group = "building"
        kind = tags["building"]
        if kind == "yes":
            for key in ["amenity", "tourism"]:
                if key in tags:
                    kind = tags[key]
                    break
        if kind != "yes":
            group += f":{kind}"
        return group
    if "area:highway" in tags:
        return f'highway:{tags["area:highway"]}'
    for key in [
        "amenity",
        "landcover",
        "leisure",
        "shop",
        "highway",
        "tourism",
        "natural",
        "waterway",
        "landuse",
    ]:
        if key in tags:
            return f"{key}:{tags[key]}"
    return None


def parse_way(tags):
    keys = tags.keys()
    for key in ["highway", "barrier", "natural"]:
        if key in keys:
            return f"{key}:{tags[key]}"
    return None


def match_to_group(label, patterns):
    for group, pattern in patterns.items():
        if re.match(pattern, label):
            return group
    return None


class Patterns:
    areas = dict(
        building="building($|:.*?)*",
        parking="amenity:parking",
        playground="leisure:(playground|pitch)",
        grass="(landuse:grass|landcover:grass|landuse:meadow|landuse:flowerbed|natural:grassland)",
        park="leisure:(park|garden|dog_park)",
        forest="(landuse:forest|natural:wood)",
        water="(natural:water|waterway:*)",
    )
    # + ways: road, path
    # + node: fountain, bicycle_parking

    ways = dict(
        fence="barrier:(fence|yes)",
        wall="barrier:(wall|retaining_wall)",
        hedge="barrier:hedge",
        kerb="barrier:kerb",
        building_outline="building($|:.*?)*",
        cycleway="highway:cycleway",
        path="highway:(pedestrian|footway|steps|path|corridor)",
        road="highway:(motorway|trunk|primary|secondary|tertiary|service|construction|track|unclassified|residential|.*_link)",
        busway="highway:busway",
        tree_row="natural:tree_row",  # maybe merge with node?
    )
    # + nodes: bollard

    nodes = dict(
        tree="natural:tree",
        stone="(natural:stone|barrier:block)",
        crossing="highway:crossing",
        lamp="highway:street_lamp",
        traffic_signal="highway:traffic_signals",
        bus_stop="highway:bus_stop",
        stop_sign="highway:stop",
        junction="highway:motorway_junction",
        bus_stop_position="public_transport:stop_position",
        gate="barrier:(gate|lift_gate|swing_gate|cycle_barrier)",
        bollard="barrier:bollard",
        shop="(shop.*?|amenity:(bank|post_office))",
        restaurant="amenity:(restaurant|fast_food)",
        bar="amenity:(cafe|bar|pub|biergarten)",
        pharmacy="amenity:pharmacy",
        fuel="amenity:fuel",
        bicycle_parking="amenity:(bicycle_parking|bicycle_rental)",
        charging_station="amenity:charging_station",
        parking_entrance="amenity:parking_entrance",
        atm="amenity:atm",
        toilets="amenity:toilets",
        vending_machine="amenity:vending_machine",
        fountain="amenity:fountain",
        waste_basket="amenity:(waste_basket|waste_disposal)",
        bench="amenity:bench",
        post_box="amenity:post_box",
        artwork="tourism:artwork",
        recycling="amenity:recycling",
        give_way="highway:give_way",
        clock="amenity:clock",
        fire_hydrant="emergency:fire_hydrant",
        pole="man_made:(flagpole|utility_pole)",
        street_cabinet="man_made:street_cabinet",
    )
    # + ways: kerb


class Groups:
    areas = list(Patterns.areas)
    ways = list(Patterns.ways)
    nodes = list(Patterns.nodes)


def group_elements(osm: OSMData):
    elem2group = {
        "area": {},
        "way": {},
        "node": {},
    }

    for node in filter(filter_node, osm.nodes.values()):
        label = parse_node(node.tags)
        if label is None:
            continue
        group = match_to_group(label, Patterns.nodes)
        if group is None:
            group = match_to_group(label, Patterns.ways)
        if group is None:
            continue  # missing
        elem2group["node"][node.id_] = group

    for way in filter(filter_way, osm.ways.values()):
        label = parse_way(way.tags)
        if label is None:
            continue
        group = match_to_group(label, Patterns.ways)
        if group is None:
            group = match_to_group(label, Patterns.nodes)
        if group is None:
            continue  # missing
        elem2group["way"][way.id_] = group

    for area in filter(filter_area, osm.ways.values()):
        label = parse_area(area.tags)
        if label is None:
            continue
        group = match_to_group(label, Patterns.areas)
        if group is None:
            group = match_to_group(label, Patterns.ways)
        if group is None:
            group = match_to_group(label, Patterns.nodes)
        if group is None:
            continue  # missing
        elem2group["area"][area.id_] = group

    return elem2group
