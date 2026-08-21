"""Per-object metro accessibility features (ADR-0011).

Graph-distance (OSM pedestrian graph) from each object to the nearest
metro station / entrance, plus graph-distance counts in fixed radii.

ADR-0030: unreachable objects (no graph path — e.g. before the graph
fixes, snapped into a detached component) get **null**, not the retired
``1e9 m`` far-sentinel. A finite fake distance was read by the model as
"very far" and painted 'milliard-meter' artifacts over whole areas;
null is the honest "unknown" and is handled natively by CatBoost/EBM.
Empty station/entrance lists get the same null treatment.
"""

import numpy as np
import polars as pl

from kadastra.ports.road_graph import RoadGraphPort


def compute_object_metro_features(
    objects: pl.DataFrame,
    stations: pl.DataFrame,
    entrances: pl.DataFrame,
    *,
    road_graph: RoadGraphPort,
) -> pl.DataFrame:
    n = objects.height
    if n == 0:
        return objects.with_columns(
            [
                pl.lit(None, dtype=pl.Float64).alias("dist_metro_m"),
                pl.lit(None, dtype=pl.Float64).alias("dist_entrance_m"),
                pl.lit(None, dtype=pl.Int64).alias("count_stations_1km"),
                pl.lit(None, dtype=pl.Int64).alias("count_entrances_500m"),
            ]
        )

    obj_coords = [
        (float(lat), float(lon)) for lat, lon in zip(objects["lat"].to_list(), objects["lon"].to_list(), strict=True)
    ]

    if stations.is_empty():
        dist_min_stations = np.full(n, np.nan, dtype=np.float64)
        cnt_stations_1km = np.zeros(n, dtype=np.int64)
    else:
        station_coords = [
            (float(lat), float(lon))
            for lat, lon in zip(
                stations["lat"].to_list(),
                stations["lon"].to_list(),
                strict=True,
            )
        ]
        d = road_graph.distance_matrix_m(obj_coords, station_coords)
        dist_min_stations = d.min(axis=1)
        dist_min_stations = np.where(np.isinf(dist_min_stations), np.nan, dist_min_stations)
        cnt_stations_1km = (d < 1000.0).sum(axis=1).astype(np.int64)

    if entrances.is_empty():
        dist_min_entrances = np.full(n, np.nan, dtype=np.float64)
        cnt_entrances_500m = np.zeros(n, dtype=np.int64)
    else:
        entrance_coords = [
            (float(lat), float(lon))
            for lat, lon in zip(
                entrances["lat"].to_list(),
                entrances["lon"].to_list(),
                strict=True,
            )
        ]
        d = road_graph.distance_matrix_m(obj_coords, entrance_coords)
        dist_min_entrances = d.min(axis=1)
        dist_min_entrances = np.where(np.isinf(dist_min_entrances), np.nan, dist_min_entrances)
        cnt_entrances_500m = (d < 500.0).sum(axis=1).astype(np.int64)

    return objects.with_columns(
        [
            pl.Series("dist_metro_m", dist_min_stations).fill_nan(None),
            pl.Series("dist_entrance_m", dist_min_entrances).fill_nan(None),
            pl.Series("count_stations_1km", cnt_stations_1km),
            pl.Series("count_entrances_500m", cnt_entrances_500m),
        ]
    )
