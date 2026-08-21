"""Road graph port — pairwise shortest-path distances over a graph.

Methodology compliance: per ADR-0010 / info/grid-rationale.md §7.2,
all distance ZOF must be computed over an OSM-derived graph rather
than as straight-line haversine. This port lets use cases consume
graph distances without knowing how the graph is built or stored.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class RoadGraphPort(Protocol):
    def distance_matrix_m(
        self,
        from_coords: list[tuple[float, float]],
        to_coords: list[tuple[float, float]],
    ) -> np.ndarray:
        """Pairwise shortest-path distances in meters.

        Each input coord is (lat, lon) in WGS84. The implementation
        snaps each query coord to the nearest graph node and adds the
        haversine snap distance to the path length on both ends, so
        callers get an end-to-end estimate of "real" walking/driving
        distance even when query points lie off the graph.

        Returns
        -------
        np.ndarray
            Shape ``(len(from_coords), len(to_coords))``. Disconnected
            pairs are reported as ``float('inf')``.
        """
        ...

    def nearest_distance_m(
        self,
        from_coords: list[tuple[float, float]],
        to_coords: list[tuple[float, float]],
    ) -> np.ndarray:
        """Distance from each ``from`` coord to its nearest ``to`` coord.

        Same snap semantics as :meth:`distance_matrix_m`, but returns a
        length-``len(from_coords)`` vector of distances to the closest
        target instead of the full pairwise matrix. Implementations
        should use a single multi-source shortest-path pass (one
        Dijkstra from all targets) so this stays ``O((V+E) log V)``
        regardless of how many targets there are.
        """
        ...

    def snap_node(self, coord: tuple[float, float]) -> tuple[int, float]:
        """Snap a ``(lat, lon)`` coord to the nearest graph node.

        Returns ``(node_id, snap_distance_m)`` where the snap distance is
        true haversine metres between the coord and the node. Node ids
        are implementation-defined but stable for the lifetime of the
        graph instance, so callers may pre-snap POI sets once and reuse
        them across many :meth:`reachable_nodes_within_m` queries.
        """
        ...

    def node_coord(self, node_id: int) -> tuple[float, float]:
        """Return the ``(lat, lon)`` of a node id from :meth:`snap_node`."""
        ...

    def reachable_nodes_within_m(
        self,
        from_coord: tuple[float, float],
        cutoff_m: float,
    ) -> dict[int, float]:
        """Nodes reachable from ``from_coord`` within ``cutoff_m``.

        The coord is snapped first; the returned map is
        ``node_id → shortest distance in metres`` *including the source
        snap cost* (so a caller adding a target's own snap distance gets
        the end-to-end estimate, same convention as
        :meth:`distance_matrix_m`). When the snap alone exceeds the
        cutoff the map is empty. Used by the ADR-0024 walking-isochrone
        enrichment.
        """
        ...
