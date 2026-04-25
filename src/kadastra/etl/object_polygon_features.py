"""Poly-area buffer features.

For each (polygon layer, radius) pair, attach a column
``{layer}_share_{R}m`` whose value is the share of the buffer-circle
area at the object covered by polygons of that layer (range [0, 1]).

ADR-0014 explains the rationale (UTM 39N projection, STRtree-indexed
shapely intersections) and the four starter layers (water, park,
industrial, cemetery).
"""

from __future__ import annotations

import polars as pl
from shapely.geometry.base import BaseGeometry


def compute_object_polygon_features(
    objects: pl.DataFrame,
    *,
    polygons_by_layer: dict[str, list[BaseGeometry]],
    radii_m: list[int],
) -> pl.DataFrame:
    raise NotImplementedError
