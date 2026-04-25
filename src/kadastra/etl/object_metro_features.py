import numpy as np
import polars as pl

from kadastra.etl.haversine import EARTH_RADIUS_METERS

_FAR_SENTINEL_M = 1.0e9


def _haversine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise haversine distance in meters; a is (n, 2), b is (m, 2); returns (n, m)."""
    rlat1 = np.radians(a[:, 0:1])
    rlon1 = np.radians(a[:, 1:2])
    rlat2 = np.radians(b[:, 0:1].T)
    rlon2 = np.radians(b[:, 1:2].T)
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    h = np.sin(dlat / 2) ** 2 + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_METERS * np.arcsin(np.sqrt(h))


def compute_object_metro_features(
    objects: pl.DataFrame,
    stations: pl.DataFrame,
    entrances: pl.DataFrame,
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

    obj_coords = objects.select(["lat", "lon"]).to_numpy()

    if stations.is_empty():
        dist_min_stations = np.full(n, _FAR_SENTINEL_M, dtype=np.float64)
        cnt_stations_1km = np.zeros(n, dtype=np.int64)
    else:
        d = _haversine_matrix(obj_coords, stations.select(["lat", "lon"]).to_numpy())
        dist_min_stations = d.min(axis=1)
        cnt_stations_1km = (d < 1000.0).sum(axis=1).astype(np.int64)

    if entrances.is_empty():
        dist_min_entrances = np.full(n, _FAR_SENTINEL_M, dtype=np.float64)
        cnt_entrances_500m = np.zeros(n, dtype=np.int64)
    else:
        d = _haversine_matrix(obj_coords, entrances.select(["lat", "lon"]).to_numpy())
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
