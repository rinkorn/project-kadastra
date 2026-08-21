"""Water-crossing filter for road-graph edges (ADR-0030).

Stub for the TDD cycle — tests land first, implementation follows.
"""

from __future__ import annotations

import polars as pl
from shapely.geometry.base import BaseGeometry

DEFAULT_MAX_WATER_CROSSING_M = 30.0


def filter_water_crossing_edges(
    edges: pl.DataFrame,
    water_polygons: list[BaseGeometry],
    *,
    max_crossing_m: float = DEFAULT_MAX_WATER_CROSSING_M,
) -> pl.DataFrame:
    raise NotImplementedError
