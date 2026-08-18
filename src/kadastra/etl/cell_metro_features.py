"""ADR-0011 → ADR-0027: per-cell metro accessibility (graph distance).

Re-points ``compute_object_metro_features`` to the cell grid: walking
distance (OSM pedestrian graph) from each cell centre to the nearest
station/entrance, plus graph-distance counts. Supersedes the legacy
haversine ``compute_metro_features`` for the object pipeline — the
methodology (§7.2) treats "nearest" as by travel path, not geometry.
"""

from __future__ import annotations

import polars as pl

from kadastra.etl.h3_coverage import h3_cells_to_latlng
from kadastra.etl.object_metro_features import compute_object_metro_features
from kadastra.ports.road_graph import RoadGraphPort


def compute_cell_metro_features(
    cells: pl.DataFrame,
    stations: pl.DataFrame,
    entrances: pl.DataFrame,
    *,
    road_graph: RoadGraphPort,
) -> pl.DataFrame:
    """Append metro columns measured at each cell centre.

    ``cells`` must carry an ``h3_index`` column. Returns ``h3_index`` plus
    ``dist_metro_m``, ``dist_entrance_m``, ``count_stations_1km``,
    ``count_entrances_500m`` (graph-distance based); no lat/lon.
    """
    if cells.is_empty():
        return cells.with_columns(
            [
                pl.lit(None, dtype=pl.Float64).alias("dist_metro_m"),
                pl.lit(None, dtype=pl.Float64).alias("dist_entrance_m"),
                pl.lit(None, dtype=pl.Int64).alias("count_stations_1km"),
                pl.lit(None, dtype=pl.Int64).alias("count_entrances_500m"),
            ]
        )

    centers = h3_cells_to_latlng(cells["h3_index"].to_list())
    points = cells.with_columns(
        [
            pl.Series("lat", [lat for lat, _ in centers], dtype=pl.Float64),
            pl.Series("lon", [lon for _, lon in centers], dtype=pl.Float64),
        ]
    )
    enriched = compute_object_metro_features(points, stations, entrances, road_graph=road_graph)
    return enriched.drop(["lat", "lon"])
