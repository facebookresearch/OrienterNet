# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from lxml import etree
import numpy as np

from ..utils.geo import BoundaryBox, Projection

METERS_PATTERN: re.Pattern = re.compile("^(?P<value>\\d*\\.?\\d*)\\s*m$")
KILOMETERS_PATTERN: re.Pattern = re.compile("^(?P<value>\\d*\\.?\\d*)\\s*km$")
MILES_PATTERN: re.Pattern = re.compile("^(?P<value>\\d*\\.?\\d*)\\s*mi$")


def parse_float(string: str) -> Optional[float]:
    """Parse string representation of a float or integer value."""
    try:
        return float(string)
    except (TypeError, ValueError):
        return None


@dataclass(eq=False)
class OSMElement:
    """
    Something with tags (string to string mapping).
    """

    id_: int
    tags: Dict[str, str]

    def get_float(self, key: str) -> Optional[float]:
        """Parse float from tag value."""
        if key in self.tags:
            return parse_float(self.tags[key])
        return None

    def get_length(self, key: str) -> Optional[float]:
        """Get length in meters."""
        if key not in self.tags:
            return None

        value: str = self.tags[key]

        float_value: float = parse_float(value)
        if float_value is not None:
            return float_value

        for pattern, ratio in [
            (METERS_PATTERN, 1.0),
            (KILOMETERS_PATTERN, 1000.0),
            (MILES_PATTERN, 1609.344),
        ]:
            matcher: re.Match = pattern.match(value)
            if matcher:
                float_value: float = parse_float(matcher.group("value"))
                if float_value is not None:
                    return float_value * ratio

        return None

    def __hash__(self) -> int:
        return self.id_


@dataclass(eq=False)
class OSMNode(OSMElement):
    """
    OpenStreetMap node.

    See https://wiki.openstreetmap.org/wiki/Node
    """

    geo: np.ndarray
    visible: Optional[str] = None
    xy: Optional[np.ndarray] = None

    @classmethod
    def from_dict(cls, structure: Dict[str, Any]) -> "OSMNode":
        """
        Parse node from Overpass-like structure.

        :param structure: input structure
        """
        return cls(
            structure["id"],
            structure.get("tags", {}),
            geo=np.array((structure["lat"], structure["lon"])),
            visible=structure.get("visible"),
        )


@dataclass(eq=False)
class OSMWay(OSMElement):
    """
    OpenStreetMap way.

    See https://wiki.openstreetmap.org/wiki/Way
    """

    nodes: Optional[List[OSMNode]] = field(default_factory=list)
    visible: Optional[str] = None

    @classmethod
    def from_dict(
        cls, structure: Dict[str, Any], nodes: Dict[int, OSMNode]
    ) -> "OSMWay":
        """
        Parse way from Overpass-like structure.

        :param structure: input structure
        :param nodes: node structure
        """
        return cls(
            structure["id"],
            structure.get("tags", {}),
            [nodes[x] for x in structure["nodes"]],
            visible=structure.get("visible"),
        )

    def is_cycle(self) -> bool:
        """Is way a cycle way or an area boundary."""
        return self.nodes[0] == self.nodes[-1]

    def __repr__(self) -> str:
        return f"Way <{self.id_}> {self.nodes}"


@dataclass
class OSMMember:
    """
    Member of OpenStreetMap relation.
    """

    type_: str
    ref: int
    role: str


@dataclass(eq=False)
class OSMRelation(OSMElement):
    """
    OpenStreetMap relation.

    See https://wiki.openstreetmap.org/wiki/Relation
    """

    members: Optional[List[OSMMember]]
    visible: Optional[str] = None

    @classmethod
    def from_dict(cls, structure: Dict[str, Any]) -> "OSMRelation":
        """
        Parse relation from Overpass-like structure.

        :param structure: input structure
        """
        return cls(
            structure["id"],
            structure["tags"],
            [OSMMember(x["type"], x["ref"], x["role"]) for x in structure["members"]],
            visible=structure.get("visible"),
        )


class OSMData:
    """
    The whole OpenStreetMap information about nodes, ways, and relations.
    """

    def __init__(self) -> None:
        self.nodes: Dict[int, OSMNode] = {}
        self.ways: Dict[int, OSMWay] = {}
        self.relations: Dict[int, OSMRelation] = {}
        self.box: BoundaryBox = None

    @classmethod
    def from_dict(cls, structure: Dict[str, Any]):
        data = cls()
        bounds = structure.get("bounds")
        if bounds is not None:
            data.box = BoundaryBox(
                np.array([bounds["minlat"], bounds["minlon"]]),
                np.array([bounds["maxlat"], bounds["maxlon"]]),
            )

        for element in structure["elements"]:
            if element["type"] == "node":
                node = OSMNode.from_dict(element)
                data.add_node(node)
        for element in structure["elements"]:
            if element["type"] == "way":
                way = OSMWay.from_dict(element, data.nodes)
                data.add_way(way)
        for element in structure["elements"]:
            if element["type"] == "relation":
                relation = OSMRelation.from_dict(element)
                data.add_relation(relation)

        return data

    @classmethod
    def from_json(cls, path: Path):
        with path.open() as fid:
            structure = json.load(fid)
        return cls.from_dict(structure)

    @classmethod
    def from_xml(cls, path: Path):
        root = etree.parse(str(path)).getroot()
        structure = {"elements": []}
        from tqdm import tqdm

        for elem in tqdm(root):
            if elem.tag == "bounds":
                structure["bounds"] = {
                    k: float(elem.attrib[k])
                    for k in ("minlon", "minlat", "maxlon", "maxlat")
                }
            elif elem.tag in {"node", "way", "relation"}:
                if elem.tag == "node":
                    item = {
                        "id": int(elem.attrib["id"]),
                        "lat": float(elem.attrib["lat"]),
                        "lon": float(elem.attrib["lon"]),
                        "visible": elem.attrib.get("visible"),
                        "tags": {
                            x.attrib["k"]: x.attrib["v"] for x in elem if x.tag == "tag"
                        },
                    }
                elif elem.tag == "way":
                    item = {
                        "id": int(elem.attrib["id"]),
                        "visible": elem.attrib.get("visible"),
                        "tags": {
                            x.attrib["k"]: x.attrib["v"] for x in elem if x.tag == "tag"
                        },
                        "nodes": [int(x.attrib["ref"]) for x in elem if x.tag == "nd"],
                    }
                elif elem.tag == "relation":
                    item = {
                        "id": int(elem.attrib["id"]),
                        "visible": elem.attrib.get("visible"),
                        "tags": {
                            x.attrib["k"]: x.attrib["v"] for x in elem if x.tag == "tag"
                        },
                        "members": [
                            {
                                "type": x.attrib["type"],
                                "ref": int(x.attrib["ref"]),
                                "role": x.attrib["role"],
                            }
                            for x in elem
                            if x.tag == "member"
                        ],
                    }
                item["type"] = elem.tag
                structure["elements"].append(item)
            elem.clear()
        del root
        return cls.from_dict(structure)

    @classmethod
    def from_file(cls, path: Path):
        ext = path.suffix
        if ext == ".json":
            return cls.from_json(path)
        elif ext in {".osm", ".xml"}:
            return cls.from_xml(path)
        else:
            raise ValueError(f"Unknown extension for {path}")

    def add_node(self, node: OSMNode):
        """Add node and update map parameters."""
        if node.id_ in self.nodes:
            raise ValueError(f"Node with duplicate id {node.id_}.")
        self.nodes[node.id_] = node

    def add_way(self, way: OSMWay):
        """Add way and update map parameters."""
        if way.id_ in self.ways:
            raise ValueError(f"Way with duplicate id {way.id_}.")
        self.ways[way.id_] = way

    def add_relation(self, relation: OSMRelation):
        """Add relation and update map parameters."""
        if relation.id_ in self.relations:
            raise ValueError(f"Relation with duplicate id {relation.id_}.")
        self.relations[relation.id_] = relation

    def add_xy_to_nodes(self, proj: Projection):
        nodes = list(self.nodes.values())
        if len(nodes) == 0:
            return
        geos = np.stack([n.geo for n in nodes], 0)
        if proj.bounds is not None:
            # For some reasons few nodes are sometimes very far off the initial bbox.
            valid = proj.bounds.contains(geos)
            if valid.mean() < 0.9:
                print("Many nodes are out of the projection bounds.")
            xys = np.zeros_like(geos)
            xys[valid] = proj.project(geos[valid])
        else:
            xys = proj.project(geos)
        for xy, node in zip(xys, nodes):
            node.xy = xy
