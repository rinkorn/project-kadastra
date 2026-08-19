"""ADR-0013 → ADR-0027: per-cell zonal density (point counts in buffer).

Re-points ``compute_object_zonal_features`` to the cell grid: count of
layer points within each radius of each cell centre. Self-free by
construction — a cell is a location, not an object, so there is no
"self" to exclude (ADR-0027). Density is a property of the place.
"""

from __future__ import annotations

import polars as pl

from kadastra.etl.h3_coverage import h3_cells_to_latlng
from kadastra.etl.object_zonal_features import compute_object_zonal_features


def compute_cell_zonal_features(
    cells: pl.DataFrame,
    *,
    layers: dict[str, pl.DataFrame],
    radii_m: list[int],
) -> pl.DataFrame:
    """Append ``{layer}_within_{R}m`` columns measured at each cell centre.

    ``cells`` must carry an ``h3_index`` column. Each layer is a DataFrame
    with ``lat``/``lon``; an ``object_id`` column, if present, is ignored —
    the cell frame has no object to self-exclude. Returns the frame with
    ``h3_index`` plus one count column per (layer, radius); no lat/lon.
    """
    if not layers or not radii_m:
        return cells

    if cells.is_empty():
        radii_sorted = sorted({int(r) for r in radii_m})
        return cells.with_columns(
            [pl.lit(None, dtype=pl.Int64).alias(f"{layer}_within_{r}m") for layer in layers for r in radii_sorted]
        )

    centers = h3_cells_to_latlng(cells["h3_index"].to_list())
    points = cells.with_columns(
        [
            pl.Series("lat", [lat for lat, _ in centers], dtype=pl.Float64),
            pl.Series("lon", [lon for _, lon in centers], dtype=pl.Float64),
        ]
    )
    enriched = compute_object_zonal_features(points, layers=layers, radii_m=radii_m)
    return enriched.drop(["lat", "lon"])
