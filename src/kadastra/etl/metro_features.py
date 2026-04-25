import h3
import numpy as np
import polars as pl

from kadastra.etl.haversine import EARTH_RADIUS_METERS


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


def compute_metro_features(
    coverage: pl.DataFrame,
    stations: pl.DataFrame,
    entrances: pl.DataFrame,
) -> pl.DataFrame:
    h3_indices = coverage["h3_index"].to_list()
    centers = np.array([h3.cell_to_latlng(idx) for idx in h3_indices], dtype=np.float64)

    station_coords = stations.select(["lat", "lon"]).to_numpy()
    entrance_coords = entrances.select(["lat", "lon"]).to_numpy()

    dist_stations = _haversine_matrix(centers, station_coords)
    dist_entrances = _haversine_matrix(centers, entrance_coords)

    return coverage.with_columns(
        pl.Series("dist_metro_m", dist_stations.min(axis=1)),
        pl.Series("dist_entrance_m", dist_entrances.min(axis=1)),
        pl.Series("count_stations_1km", (dist_stations < 1000.0).sum(axis=1).astype(np.int64)),
        pl.Series("count_entrances_500m", (dist_entrances < 500.0).sum(axis=1).astype(np.int64)),
    )
