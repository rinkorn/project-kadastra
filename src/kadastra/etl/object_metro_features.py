import numpy as np
import polars as pl

from kadastra.ports.road_graph import RoadGraphPort

_FAR_SENTINEL_M = 1.0e9


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
        (float(lat), float(lon))
        for lat, lon in zip(
            objects["lat"].to_list(), objects["lon"].to_list(), strict=True
        )
    ]

    if stations.is_empty():
        dist_min_stations = np.full(n, _FAR_SENTINEL_M, dtype=np.float64)
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
        cnt_stations_1km = (d < 1000.0).sum(axis=1).astype(np.int64)

    if entrances.is_empty():
        dist_min_entrances = np.full(n, _FAR_SENTINEL_M, dtype=np.float64)
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
        cnt_entrances_500m = (d < 500.0).sum(axis=1).astype(np.int64)

    return objects.with_columns(
        [
            pl.Series("dist_metro_m", dist_min_stations),
            pl.Series("dist_entrance_m", dist_min_entrances),
            pl.Series("count_stations_1km", cnt_stations_1km),
            pl.Series("count_entrances_500m", cnt_entrances_500m),
        ]
    )
