import numpy as np
import polars as pl

BASE_FLOOR_RUB_PER_M2 = 1_000.0
BASE_INFRA_RUB_PER_M2 = 6_000.0
ROAD_WEIGHT = 0.3
INFRA_EXPONENT = 1.2
METRO_BOOST = 0.5
NOISE_SIGMA = 2_000.0

_REQUIRED_COLUMNS = (
    "h3_index",
    "resolution",
    "building_count_apartments",
    "count_stations_1km",
    "road_length_m",
)


def compute_synthetic_target(gold: pl.DataFrame, *, seed: int = 42) -> pl.DataFrame:
    missing = [c for c in _REQUIRED_COLUMNS if c not in gold.columns]
    if missing:
        raise KeyError(f"required columns missing from gold: {missing}")

    apartments = gold["building_count_apartments"].fill_null(0).to_numpy().astype(np.float64)
    stations = gold["count_stations_1km"].fill_null(0).to_numpy().astype(np.float64)
    road_km = gold["road_length_m"].fill_null(0.0).to_numpy().astype(np.float64) / 1000.0

    infra_score = np.log1p(apartments) + ROAD_WEIGHT * np.log1p(road_km)
    metro_factor = 1.0 + METRO_BOOST * (stations > 0).astype(np.float64)

    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=NOISE_SIGMA, size=gold.height)

    target = (
        BASE_FLOOR_RUB_PER_M2
        + BASE_INFRA_RUB_PER_M2 * (infra_score**INFRA_EXPONENT) * metro_factor
        + noise
    )
    target = np.clip(target, a_min=0.0, a_max=None)

    return gold.with_columns(pl.Series("synthetic_target_rub_per_m2", target))
