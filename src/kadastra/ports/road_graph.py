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
