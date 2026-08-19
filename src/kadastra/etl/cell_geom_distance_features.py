"""ADR-0027: per-cell distance to the nearest geometry of each layer.

Re-points ``compute_object_geom_distance_features`` to the cell grid:
instead of measuring from each object's own coordinate, measure from each
cell's centre (its reference point), producing Слой 1 location ЦОФ keyed
by ``h3_index``. A point object later inherits these values by joining to
its cell (weighted by overlap for large objects).
"""

from __future__ import annotations

import polars as pl
from shapely.geometry.base import BaseGeometry

from kadastra.etl.h3_coverage import h3_cells_to_latlng
from kadastra.etl.object_geom_distance_features import (
    compute_object_geom_distance_features,
)


def compute_cell_geom_distance_features(
    cells: pl.DataFrame,
    *,
    geometries_by_layer: dict[str, list[BaseGeometry]],
) -> pl.DataFrame:
    """Append ``dist_to_<layer>_m`` columns measured from each cell centre.

    ``cells`` must carry an ``h3_index`` column. The returned frame keeps
    ``h3_index`` plus one distance column per layer (no lat/lon). Empty
    layers produce all-null columns; empty input emits the null columns
    with the expected names so downstream schema is stable.
    """
    if not geometries_by_layer:
        return cells

    if cells.is_empty():
        return cells.with_columns(
            [pl.lit(None, dtype=pl.Float64).alias(f"dist_to_{layer}_m") for layer in geometries_by_layer]
        )

    centers = h3_cells_to_latlng(cells["h3_index"].to_list())
    points = cells.with_columns(
        [
            pl.Series("lat", [lat for lat, _ in centers], dtype=pl.Float64),
            pl.Series("lon", [lon for _, lon in centers], dtype=pl.Float64),
        ]
    )
    enriched = compute_object_geom_distance_features(points, geometries_by_layer=geometries_by_layer)
    return enriched.drop(["lat", "lon"])
