# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from typing import List, Optional, Tuple

import numpy as np
import rtree

from ..utils.geo import BoundaryBox
from .data import MapData
from .reader import OSMData, OSMNode, OSMWay


class OSMIndex:
    def __init__(
        self,
        osm: OSMData,
        node_ids: Optional[List[int]] = None,
        way_ids: Optional[List[int]] = None,
    ):
        self.index_nodes = rtree.index.Index()
        if node_ids is None:
            node_ids = osm.nodes.keys()
        for id_ in node_ids:
            self.index_nodes.insert(id_, tuple(osm.nodes[id_].xy) * 2)

        self.index_ways = rtree.index.Index()
        if way_ids is None:
            way_ids = osm.ways.keys()
        for id_ in way_ids:
            xy = np.stack([node.xy for node in osm.ways[id_].nodes])
            bbox = tuple(np.r_[xy.min(0), xy.max(0)])
            self.index_ways.insert(id_, bbox)

        self.osm = osm

    def query(self, bbox: BoundaryBox) -> Tuple[List[OSMNode], List[OSMWay]]:
        query = tuple(np.r_[bbox.min_, bbox.max_])
        node_ids = self.index_nodes.intersection(query)
        nodes = [self.osm.nodes[i] for i in node_ids]
        way_ids = self.index_ways.intersection(query)
        ways = [self.osm.ways[i] for i in way_ids]
        return nodes, ways


class MapIndex:
    def __init__(
        self,
        data: MapData,
    ):
        self.index_nodes = rtree.index.Index()
        for i, node in data.nodes.items():
            self.index_nodes.insert(i, tuple(node.xy) * 2)

        self.index_lines = rtree.index.Index()
        for i, line in data.lines.items():
            bbox = tuple(np.r_[line.xy.min(0), line.xy.max(0)])
            self.index_lines.insert(i, bbox)

        self.index_areas = rtree.index.Index()
        for i, area in data.areas.items():
            xy = np.concatenate(area.outers + area.inners)
            bbox = tuple(np.r_[xy.min(0), xy.max(0)])
            self.index_areas.insert(i, bbox)

        self.data = data

    def query(self, bbox: BoundaryBox) -> Tuple[List[OSMNode], List[OSMWay]]:
        query = tuple(np.r_[bbox.min_, bbox.max_])
        ret = []
        for x in ["nodes", "lines", "areas"]:
            ids = getattr(self, "index_" + x).intersection(query)
            ret.append([getattr(self.data, x)[i] for i in ids])
        return tuple(ret)
