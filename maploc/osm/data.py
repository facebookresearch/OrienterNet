# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import geopandas as gpd

import numpy as np
from shapely import (
    get_exterior_ring,
    get_interior_ring,
    get_num_interior_rings,
    get_parts,
)
from shapely.ops import transform

from ..utils.geo import Projection

from .parser import (
    filter_area,
    filter_node,
    filter_way,
    match_to_group,
    parse_area,
    parse_node,
    parse_way,
    Patterns,
)
from .reader import OSMData, OSMNode, OSMRelation, OSMWay

logger = logging.getLogger(__name__)


def glue(ways: List[OSMWay]) -> List[List[OSMNode]]:
    result: List[List[OSMNode]] = []
    to_process: Set[Tuple[OSMNode]] = set()

    for way in ways:
        if way.is_cycle():
            result.append(way.nodes)
        else:
            to_process.add(tuple(way.nodes))

    while to_process:
        nodes: List[OSMNode] = list(to_process.pop())
        glued: Optional[List[OSMNode]] = None
        other_nodes: Optional[Tuple[OSMNode]] = None

        for other_nodes in to_process:
            glued = try_to_glue(nodes, list(other_nodes))
            if glued is not None:
                break

        if glued is not None:
            to_process.remove(other_nodes)
            if is_cycle(glued):
                result.append(glued)
            else:
                to_process.add(tuple(glued))
        else:
            result.append(nodes)

    return result


def is_cycle(nodes: List[OSMNode]) -> bool:
    """Is way a cycle way or an area boundary."""
    return nodes[0] == nodes[-1]


def try_to_glue(nodes: List[OSMNode], other: List[OSMNode]) -> Optional[List[OSMNode]]:
    """Create new combined way if ways share endpoints."""
    if nodes[0] == other[0]:
        return list(reversed(other[1:])) + nodes
    if nodes[0] == other[-1]:
        return other[:-1] + nodes
    if nodes[-1] == other[-1]:
        return nodes + list(reversed(other[:-1]))
    if nodes[-1] == other[0]:
        return nodes + other[1:]
    return None


def multipolygon_from_relation(rel: OSMRelation, osm: OSMData):
    inner_ways = []
    outer_ways = []
    for member in rel.members:
        if member.type_ == "way":
            if member.role == "inner":
                if member.ref in osm.ways:
                    inner_ways.append(osm.ways[member.ref])
            elif member.role == "outer":
                if member.ref in osm.ways:
                    outer_ways.append(osm.ways[member.ref])
            else:
                logger.warning(f'Unknown member role "{member.role}".')
    if outer_ways:
        inners_path = glue(inner_ways)
        outers_path = glue(outer_ways)
        return inners_path, outers_path


@dataclass
class MapElement:
    id_: int
    label: str
    group: str
    tags: Optional[Dict[str, str]]


@dataclass
class MapNode(MapElement):
    xy: np.ndarray

    @classmethod
    def from_osm(cls, node: OSMNode, label: str, group: str):
        return cls(
            node.id_,
            label,
            group,
            node.tags,
            xy=node.xy,
        )

    @classmethod
    def from_geojson(cls, feature, projection, label, group):
        _coords = feature.get("geometry").get("coordinates")
        lat_lon = [_coords[1], _coords[0]]

        return cls(
            feature.get("properties").get("id"),
            label,
            group,
            feature.get("properties"),
            xy=projection.project(lat_lon),
        )


@dataclass
class MapLine(MapElement):
    xy: np.ndarray

    @classmethod
    def from_osm(cls, way: OSMWay, label: str, group: str):
        xy = np.stack([n.xy for n in way.nodes])
        return cls(
            way.id_,
            label,
            group,
            way.tags,
            xy=xy,
        )
        xy = [[_[1], _[0]] for _ in feature.get("geometry").get("coordinates")]
        return cls(
            feature.get("properties").get("id"),
            label,
            group,
            feature.get("properties"),
            xy=projection.project(xy),
        )


@dataclass
class MapArea(MapElement):
    outers: List[np.ndarray]
    inners: List[np.ndarray] = field(default_factory=list)

    @classmethod
    def from_relation(cls, rel: OSMRelation, label: str, group: str, osm: OSMData):
        outers_inners = multipolygon_from_relation(rel, osm)
        if outers_inners is None:
            return None
        outers, inners = outers_inners
        outers = [np.stack([n.xy for n in way]) for way in outers]
        inners = [np.stack([n.xy for n in way]) for way in inners]
        return cls(
            rel.id_,
            label,
            group,
            rel.tags,
            outers=outers,
            inners=inners,
        )

    @classmethod
    def from_way(cls, way: OSMWay, label: str, group: str):
        xy = np.stack([n.xy for n in way.nodes])
        return cls(
            way.id_,
            label,
            group,
            way.tags,
            outers=[xy],
        )


class MapData:
    def __init__(self):
        self.nodes: Dict[int, MapNode] = {}
        self.lines: Dict[int, MapLine] = {}
        self.areas: Dict[int, MapArea] = {}

    @classmethod
    def from_osm(cls, osm: OSMData):
        self = cls()

        for node in filter(filter_node, osm.nodes.values()):
            label = parse_node(node.tags)
            if label is None:
                continue
            group = match_to_group(label, Patterns.nodes)
            if group is None:
                group = match_to_group(label, Patterns.ways)
            if group is None:
                continue  # missing
            self.nodes[node.id_] = MapNode.from_osm(node, label, group)

        for way in filter(filter_way, osm.ways.values()):
            label = parse_way(way.tags)
            if label is None:
                continue
            group = match_to_group(label, Patterns.ways)
            if group is None:
                group = match_to_group(label, Patterns.nodes)
            if group is None:
                continue  # missing
            self.lines[way.id_] = MapLine.from_osm(way, label, group)

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
            self.areas[area.id_] = MapArea.from_way(area, label, group)

        for rel in osm.relations.values():
            if rel.tags.get("type") != "multipolygon":
                continue
            label = parse_area(rel.tags)
            if label is None:
                continue
            group = match_to_group(label, Patterns.areas)
            if group is None:
                group = match_to_group(label, Patterns.ways)
            if group is None:
                group = match_to_group(label, Patterns.nodes)
            if group is None:
                continue  # missing
            area = MapArea.from_relation(rel, label, group, osm)
            assert rel.id_ not in self.areas  # not sure if there can be collision
            if area is not None:
                self.areas[rel.id_] = area

        return self

    @classmethod
    def from_geodataframe(cls, gdf: gpd.GeoDataFrame, projection: Projection):
        """
        Expects a GeoDataFrame with the following columns:
        - geometry: shapely geometry
        - label: string label
        - group: string group

        It also expects the projection to be passed in and then used to convert
        from shapely geometry to the proper numpy array object.
        """
        self = cls()
        idx = 0

        # Areas
        for row_idx, row in gdf[
            gdf.geometry.apply(lambda g: g.geom_type).isin(["MultiPolygon", "Polygon"])
        ].iterrows():
            outers = []
            inners = []
            for p in get_parts(row.geometry):
                idx += 1
                exterior_ring = get_exterior_ring(p)
                yx = transform(lambda x, y: (y, x), exterior_ring).coords
                outers.append(projection.project(yx))

                # TODO: Haven't tested this anywhere where interior rings exit:
                for interior_ring_idx in range(0, get_num_interior_rings(p)):
                    interior_ring = get_interior_ring(p, interior_ring_idx)
                    yx = transform(lambda x, y: (y, x), interior_ring).coords
                    inners.append(projection.project(yx))

                self.areas[idx] = MapArea(
                    row_idx, row.label, row.group, {}, outers, inners
                )

        # Lines
        for row_idx, row in gdf[
            gdf.geometry.apply(lambda g: g.geom_type).isin(
                ["LineString", "MultiLineString"]
            )
        ].iterrows():
            for p in get_parts(row.geometry):
                idx += 1
                yx = transform(lambda x, y: (y, x), p).coords

                self.lines[idx] = MapLine(
                    row_idx, row.label, row.group, {}, projection.project(yx)
                )

        # "Nodes" (Points)
        for row_idx, row in gdf[
            gdf.geometry.apply(lambda g: g.geom_type).isin(["MultiPoint", "Point"])
        ].iterrows():
            for p in get_parts(row.geometry):
                idx += 1
                yx = transform(lambda x, y: (y, x), p).coords
                self.nodes[idx] = MapNode(
                    row_idx, row.label, row.group, {}, projection.project(yx)[0]
                )

        return self
