"""ADR-0014 → ADR-0027: per-cell share of each polygon layer in buffer.

Re-points ``compute_object_polygon_features`` to the cell grid: the share
of the buffer circle at each cell centre covered by polygons of that layer,
producing Слой 1 ``{layer}_share_{R}m`` keyed by ``h3_index``.
"""

from __future__ import annotations

import polars as pl
from shapely.geometry.base import BaseGeometry

from kadastra.etl.h3_coverage import h3_cells_to_latlng
from kadastra.etl.object_polygon_features import compute_object_polygon_features


def compute_cell_polygon_features(
    cells: pl.DataFrame,
    *,
    polygons_by_layer: dict[str, list[BaseGeometry]],
    radii_m: list[int],
) -> pl.DataFrame:
    """Append ``{layer}_share_{R}m`` columns measured at each cell centre.

    ``cells`` must carry an ``h3_index`` column. The returned frame keeps
    ``h3_index`` plus one share column per (layer, radius); no lat/lon.
    """
    if not polygons_by_layer or not radii_m:
        return cells

    if cells.is_empty():
        radii_sorted = sorted({int(r) for r in radii_m})
        return cells.with_columns(
            [
                pl.lit(None, dtype=pl.Float64).alias(f"{layer}_share_{r}m")
                for layer in polygons_by_layer
                for r in radii_sorted
            ]
        )

    centers = h3_cells_to_latlng(cells["h3_index"].to_list())
    points = cells.with_columns(
        [
            pl.Series("lat", [lat for lat, _ in centers], dtype=pl.Float64),
            pl.Series("lon", [lon for _, lon in centers], dtype=pl.Float64),
        ]
    )
    enriched = compute_object_polygon_features(points, polygons_by_layer=polygons_by_layer, radii_m=radii_m)
    return enriched.drop(["lat", "lon"])
