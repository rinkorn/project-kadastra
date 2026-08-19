"""ADR-0027: per-cell road density (sum of road length in buffer).

Re-points ``compute_object_road_features`` to the cell grid: sum of road
segment lengths within ``radius_m`` of each cell centre, producing Слой 1
``road_length_500m`` keyed by ``h3_index``.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from kadastra.etl.h3_coverage import h3_cells_to_latlng
from kadastra.etl.object_road_features import compute_object_road_features


def compute_cell_road_features(
    cells: pl.DataFrame,
    *,
    ways: list[dict[str, Any]],
    radius_m: float,
) -> pl.DataFrame:
    """Append ``road_length_500m`` measured at each cell centre."""
    if cells.is_empty():
        return cells.with_columns(pl.lit(None, dtype=pl.Float64).alias("road_length_500m"))

    centers = h3_cells_to_latlng(cells["h3_index"].to_list())
    points = cells.with_columns(
        [
            pl.Series("lat", [lat for lat, _ in centers], dtype=pl.Float64),
            pl.Series("lon", [lon for _, lon in centers], dtype=pl.Float64),
        ]
    )
    enriched = compute_object_road_features(points, ways=ways, radius_m=radius_m)
    return enriched.drop(["lat", "lon"])
