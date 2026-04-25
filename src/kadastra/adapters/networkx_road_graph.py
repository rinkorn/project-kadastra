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

import numpy as np

from kadastra.ports.road_graph import RoadGraphPort


_Coord = tuple[float, float]


class NetworkxRoadGraph(RoadGraphPort):
    @classmethod
    def from_edges(
        cls, edges: Iterable[tuple[_Coord, _Coord, float]]
    ) -> "NetworkxRoadGraph":
        raise NotImplementedError

    def distance_matrix_m(
        self,
        from_coords: list[_Coord],
        to_coords: list[_Coord],
    ) -> np.ndarray:
        raise NotImplementedError
