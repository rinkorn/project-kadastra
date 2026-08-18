"""ADR-0027: per-cell walking distance to the nearest point of each POI layer.

Generalizes the metro graph-distance pattern to any point layer: for each
cell centre, the walking distance (OSM pedestrian graph) to the nearest
point of each layer, producing Слой 1 ``walk_dist_to_<layer>_m`` keyed by
``h3_index``. Polygonal/linear layers (water, park, powerline, railway) are
out of scope — the graph is point-to-point, so "walking to a polygon edge"
is meaningless without entry points.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from kadastra.etl.h3_coverage import h3_cells_to_latlng
from kadastra.ports.road_graph import RoadGraphPort

_FAR_SENTINEL_M = 1.0e9


def compute_cell_graph_distance_features(
    cells: pl.DataFrame,
    *,
    point_layers: dict[str, pl.DataFrame],
    road_graph: RoadGraphPort,
) -> pl.DataFrame:
    """Append ``walk_dist_to_<layer>_m`` measured at each cell centre.

    ``cells`` must carry an ``h3_index`` column; each layer is a DataFrame
    with ``lat``/``lon`` points. Unreachable points yield the far sentinel
    (1e9), empty layers yield null. Returns the frame keyed by ``h3_index``
    plus one column per layer; no lat/lon.
    """
    if not point_layers:
        return cells

    if cells.is_empty():
        return cells.with_columns(
            [pl.lit(None, dtype=pl.Float64).alias(f"walk_dist_to_{layer}_m") for layer in point_layers]
        )

    centers = h3_cells_to_latlng(cells["h3_index"].to_list())
    points = cells.with_columns(
        [
            pl.Series("lat", [lat for lat, _ in centers], dtype=pl.Float64),
            pl.Series("lon", [lon for _, lon in centers], dtype=pl.Float64),
        ]
    )
    coords = [
        (float(lat), float(lon)) for lat, lon in zip(points["lat"].to_list(), points["lon"].to_list(), strict=True)
    ]

    new_columns: list[pl.Series] = []
    for layer, layer_df in point_layers.items():
        col = f"walk_dist_to_{layer}_m"
        if layer_df.is_empty():
            new_columns.append(pl.Series(col, [None] * len(coords), dtype=pl.Float64))
            continue
        layer_coords = [
            (float(lat), float(lon))
            for lat, lon in zip(layer_df["lat"].to_list(), layer_df["lon"].to_list(), strict=True)
        ]
        dist = road_graph.distance_matrix_m(coords, layer_coords)
        dist_min = np.where(np.isinf(dist.min(axis=1)), _FAR_SENTINEL_M, dist.min(axis=1))
        new_columns.append(pl.Series(col, dist_min))

    return points.with_columns(new_columns).drop(["lat", "lon"])
