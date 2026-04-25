"""NetworkX-backed implementation of RoadGraphPort.

Holds an undirected graph with edge weight ``length_m`` plus a KD-tree
over node coordinates for O(log n) snap lookups. Pairwise distances are
computed by single-source Dijkstra from each target node — N targets
means N Dijkstras, each O((V+E) log V), independent of how many source
points are queried (the typical access pattern is "many objects vs
few POIs", where this is much faster than per-pair shortest_path).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import networkx as nx
import numpy as np
import polars as pl
from scipy.spatial import cKDTree

from kadastra.etl.haversine import haversine_meters
from kadastra.ports.road_graph import RoadGraphPort

_Coord = tuple[float, float]
_EDGES_SCHEMA = ("from_lat", "from_lon", "to_lat", "to_lon", "length_m")


class NetworkxRoadGraph(RoadGraphPort):
    def __init__(self, graph: nx.Graph, node_coords: np.ndarray) -> None:
        if node_coords.ndim != 2 or node_coords.shape[1] != 2:
            raise ValueError(
                f"node_coords must have shape (n, 2); got {node_coords.shape}"
            )
        self._graph = graph
        self._node_coords = node_coords
        self._kdtree = cKDTree(node_coords) if len(node_coords) else None

    @classmethod
    def from_edges(
        cls, edges: Iterable[tuple[_Coord, _Coord, float]]
    ) -> NetworkxRoadGraph:
        coord_to_id: dict[_Coord, int] = {}
        node_coords: list[_Coord] = []
        graph: nx.Graph = nx.Graph()
        for from_coord, to_coord, length in edges:
            for c in (from_coord, to_coord):
                if c not in coord_to_id:
                    coord_to_id[c] = len(node_coords)
                    node_coords.append(c)
                    graph.add_node(coord_to_id[c])
            graph.add_edge(
                coord_to_id[from_coord],
                coord_to_id[to_coord],
                length_m=float(length),
            )
        return cls(graph, np.asarray(node_coords, dtype=np.float64))

    @classmethod
    def from_parquet(cls, path: Path) -> NetworkxRoadGraph:
        df = pl.read_parquet(path)
        missing = [c for c in _EDGES_SCHEMA if c not in df.columns]
        if missing:
            raise ValueError(
                f"road graph parquet at {path} missing columns: {missing}"
            )
        edges = [
            (
                (float(row[0]), float(row[1])),
                (float(row[2]), float(row[3])),
                float(row[4]),
            )
            for row in df.select(list(_EDGES_SCHEMA)).iter_rows()
        ]
        return cls.from_edges(edges)

    def _snap(self, lat: float, lon: float) -> tuple[int, float]:
        if self._kdtree is None:
            raise ValueError("graph has no nodes — cannot snap")
        # cKDTree returns squared euclidean on the (lat, lon) plane, which
        # is fine for *finding* the nearest node (rank-preserving in a
        # small region) but the snap distance we report must be true
        # haversine in meters.
        _, idx = self._kdtree.query([lat, lon], k=1)
        node_id = int(idx)
        node_lat, node_lon = self._node_coords[node_id]
        return node_id, haversine_meters(lat, lon, float(node_lat), float(node_lon))

    def distance_matrix_m(
        self,
        from_coords: list[_Coord],
        to_coords: list[_Coord],
    ) -> np.ndarray:
        n_from, n_to = len(from_coords), len(to_coords)
        out = np.full((n_from, n_to), np.inf, dtype=np.float64)
        if n_from == 0 or n_to == 0 or self._kdtree is None:
            return out

        from_snaps = [self._snap(lat, lon) for lat, lon in from_coords]
        to_snaps = [self._snap(lat, lon) for lat, lon in to_coords]

        for j, (to_node, to_snap_m) in enumerate(to_snaps):
            try:
                dist_map = nx.single_source_dijkstra_path_length(
                    self._graph, to_node, weight="length_m"
                )
            except nx.NodeNotFound:
                continue
            for i, (from_node, from_snap_m) in enumerate(from_snaps):
                if from_node in dist_map:
                    out[i, j] = float(
                        from_snap_m + dist_map[from_node] + to_snap_m
                    )
        return out
