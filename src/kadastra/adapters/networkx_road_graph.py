"""NetworkX-backed implementation of RoadGraphPort.

Holds an undirected graph with edge weight ``length_m`` plus a KD-tree
over node coordinates for O(log n) snap lookups. Pairwise distances are
computed by single-source Dijkstra from each target node — N targets
means N Dijkstras, each O((V+E) log V), independent of how many source
points are queried (the typical access pattern is "many objects vs
few POIs", where this is much faster than per-pair shortest_path).

ADR-0030: builds keep only the **largest connected component**. Real OSM
extracts carry hundreds of detached slivers (unlinked road stubs,
islands, dropped phantom crossings); snapping into one of them made the
object unreachable from everything and produced inf distances. With the
main component kept, ``_snap`` always lands on the routable network.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import networkx as nx
import numpy as np
import polars as pl
from scipy.spatial import cKDTree  # pyright: ignore[reportAttributeAccessIssue]

from kadastra.etl.haversine import haversine_meters
from kadastra.ports.road_graph import RoadGraphPort

logger = logging.getLogger(__name__)

_Coord = tuple[float, float]
_EDGES_SCHEMA = ("from_lat", "from_lon", "to_lat", "to_lon", "length_m")


class NetworkxRoadGraph(RoadGraphPort):
    def __init__(self, graph: nx.Graph[int], node_coords: np.ndarray) -> None:
        if node_coords.ndim != 2 or node_coords.shape[1] != 2:
            raise ValueError(f"node_coords must have shape (n, 2); got {node_coords.shape}")
        self._graph = graph
        self._node_coords = node_coords
        self._kdtree = cKDTree(node_coords) if len(node_coords) else None

    @classmethod
    def from_edges(cls, edges: Iterable[tuple[_Coord, _Coord, float]]) -> NetworkxRoadGraph:
        coord_to_id: dict[_Coord, int] = {}
        node_coords: list[_Coord] = []
        graph: nx.Graph[int] = nx.Graph()
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
        return cls._largest_connected_component(graph, node_coords)

    @classmethod
    def _largest_connected_component(
        cls,
        graph: nx.Graph[int],
        node_coords: list[_Coord],
    ) -> NetworkxRoadGraph:
        """Keep only the largest connected component (ADR-0030).

        Node ids are re-indexed contiguously so ``node_coords`` stays a
        plain positional array. Dropped nodes/edges are logged."""
        if graph.number_of_nodes() == 0:
            return cls(graph, np.asarray(node_coords, dtype=np.float64).reshape(0, 2))
        components = list(nx.connected_components(graph))
        if len(components) == 1:
            return cls(graph, np.asarray(node_coords, dtype=np.float64))

        largest = max(components, key=len)
        sub = graph.subgraph(largest)
        dropped_nodes = graph.number_of_nodes() - sub.number_of_nodes()
        dropped_edges = graph.number_of_edges() - sub.number_of_edges()
        logger.info(
            "road graph: keeping largest connected component "
            "(%d of %d nodes, %d of %d edges); dropped %d nodes / %d edges in %d small components",
            sub.number_of_nodes(),
            graph.number_of_nodes(),
            sub.number_of_edges(),
            graph.number_of_edges(),
            dropped_nodes,
            dropped_edges,
            len(components) - 1,
        )
        old_ids = list(sub.nodes)
        mapping = {old: new for new, old in enumerate(old_ids)}
        new_graph: nx.Graph[int] = nx.relabel_nodes(sub, mapping)
        new_coords = np.asarray([node_coords[old] for old in old_ids], dtype=np.float64)
        return cls(new_graph, new_coords)

    @classmethod
    def from_parquet(cls, path: Path) -> NetworkxRoadGraph:
        df = pl.read_parquet(path)
        missing = [c for c in _EDGES_SCHEMA if c not in df.columns]
        if missing:
            raise ValueError(f"road graph parquet at {path} missing columns: {missing}")
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

    def snap_node(self, coord: _Coord) -> tuple[int, float]:
        return self._snap(coord[0], coord[1])

    def node_coord(self, node_id: int) -> _Coord:
        lat, lon = self._node_coords[node_id]
        return float(lat), float(lon)

    def reachable_nodes_within_m(self, from_coord: _Coord, cutoff_m: float) -> dict[int, float]:
        if cutoff_m < 0:
            raise ValueError(f"cutoff_m must be non-negative; got {cutoff_m}")
        node, snap_m = self._snap(from_coord[0], from_coord[1])
        if snap_m > cutoff_m:
            return {}
        dists = nx.single_source_dijkstra_path_length(
            self._graph,
            node,
            cutoff=cutoff_m - snap_m,
            weight="length_m",
        )
        return {n: float(d + snap_m) for n, d in dists.items()}

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
                dist_map = nx.single_source_dijkstra_path_length(self._graph, to_node, weight="length_m")
            except nx.NodeNotFound:
                continue
            for i, (from_node, from_snap_m) in enumerate(from_snaps):
                if from_node in dist_map:
                    out[i, j] = float(from_snap_m + dist_map[from_node] + to_snap_m)
        return out

    def nearest_distance_m(
        self,
        from_coords: list[_Coord],
        to_coords: list[_Coord],
    ) -> np.ndarray:
        """Distance from each ``from`` coord to its nearest ``to`` coord.

        Single multi-source Dijkstra over the whole graph — one pass from
        a virtual super-source that connects every target node with its
        snap offset as the edge weight. ``O((V+E) log V)`` per call,
        independent of the target count, versus ``N`` full Dijkstras in
        :meth:`distance_matrix_m`. This is the right call when callers
        only need the nearest target (e.g. ``walk_dist_to_<layer>_m``),
        not per-target distances.
        """
        n_from = len(from_coords)
        out = np.full(n_from, np.inf, dtype=np.float64)
        if n_from == 0 or not to_coords or self._kdtree is None:
            return out

        from_snaps = [self._snap(lat, lon) for lat, lon in from_coords]
        # Dedupe target nodes, keeping the smallest snap offset — several
        # POIs can snap to the same node, and the nearest is the closest.
        source_offset: dict[int, float] = {}
        for lat, lon in to_coords:
            node, snap_m = self._snap(lat, lon)
            source_offset[node] = min(source_offset.get(node, float("inf")), float(snap_m))

        super_id = max(self._graph.nodes) + 1
        self._graph.add_node(super_id)
        for node, snap_m in source_offset.items():
            self._graph.add_edge(super_id, node, length_m=snap_m)
        try:
            dist_map = nx.single_source_dijkstra_path_length(self._graph, super_id, weight="length_m")
        finally:
            self._graph.remove_node(super_id)

        for i, (from_node, from_snap_m) in enumerate(from_snaps):
            if from_node in dist_map:
                out[i] = float(from_snap_m + dist_map[from_node])
        return out
