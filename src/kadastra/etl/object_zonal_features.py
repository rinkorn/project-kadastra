"""Zonal density features at multiple radii.

For each (point layer, radius) pair, attach a column
``{layer}_within_{R}m`` with the number of layer points within R
meters of each object (haversine distance). ADR-0013 explains
the radii choice (100/300/500/800 m), why haversine instead of
graph distance, and how self-exclusion works for self-class
density layers.
"""

from __future__ import annotations

import polars as pl


def compute_object_zonal_features(
    objects: pl.DataFrame,
    *,
    layers: dict[str, pl.DataFrame],
    radii_m: list[int],
) -> pl.DataFrame:
    raise NotImplementedError
