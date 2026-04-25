import h3
import numpy as np
import polars as pl

from kadastra.etl.haversine import EARTH_RADIUS_METERS

KAZAN_LAT = 55.7887
KAZAN_LON = 49.1221

BASE_PRICE_RUB_PER_M2 = 80_000.0
DECAY_KM = 30.0
METRO_BOOST = 0.5
BUILDINGS_BOOST = 0.1
NOISE_SIGMA = 5_000.0

_REQUIRED_COLUMNS = ("h3_index", "resolution", "building_count", "count_stations_1km")


def _haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: float, lon2: float) -> np.ndarray:
    rlat1 = np.radians(lat1)
    rlat2 = np.radians(lat2)
    dlat = rlat2 - rlat1
    dlon = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon / 2) ** 2
    return 2 * (EARTH_RADIUS_METERS / 1000.0) * np.arcsin(np.sqrt(a))


def compute_synthetic_target(gold: pl.DataFrame, *, seed: int = 42) -> pl.DataFrame:
    missing = [c for c in _REQUIRED_COLUMNS if c not in gold.columns]
    if missing:
        raise KeyError(f"required columns missing from gold: {missing}")

    h3_indices = gold["h3_index"].to_list()
    centers = [h3.cell_to_latlng(h) for h in h3_indices]
    lats = np.array([lat for lat, _ in centers], dtype=np.float64)
    lons = np.array([lon for _, lon in centers], dtype=np.float64)

    dist_km = _haversine_km(lats, lons, KAZAN_LAT, KAZAN_LON)

    building_count = gold["building_count"].to_numpy().astype(np.float64)
    count_stations_1km = gold["count_stations_1km"].to_numpy().astype(np.float64)

    decay = np.exp(-dist_km / DECAY_KM)
    metro_factor = 1.0 + METRO_BOOST * (count_stations_1km > 0).astype(np.float64)
    buildings_factor = 1.0 + BUILDINGS_BOOST * np.log1p(building_count)

    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=NOISE_SIGMA, size=len(h3_indices))

    target = BASE_PRICE_RUB_PER_M2 * decay * metro_factor * buildings_factor + noise
    target = np.clip(target, a_min=0.0, a_max=None)

    return gold.with_columns(
        pl.Series("kazan_distance_km", dist_km),
        pl.Series("synthetic_target_rub_per_m2", target),
    )
