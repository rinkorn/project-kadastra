"""ADR-0019: per-object distance to the nearest polygon of each layer.

Stub — implementation in a follow-up commit per dev-rules TDD cycle."""

from __future__ import annotations

import polars as pl
from shapely.geometry.base import BaseGeometry


def compute_object_poly_distance_features(
    objects: pl.DataFrame,
    *,
    polygons_by_layer: dict[str, list[BaseGeometry]],
) -> pl.DataFrame:
    """Append ``dist_to_<layer>_m`` columns: distance in metres
    (EPSG:32639 UTM-39N) from each object to the nearest polygon of
    each layer. Distance is 0.0 when the object is inside a polygon.

    Empty layers produce all-null columns. Empty input frames produce
    empty columns with the expected names so downstream schema is
    stable across asset-class slices.
    """
    raise NotImplementedError
