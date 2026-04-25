import numpy as np
import polars as pl

from kadastra.domain.asset_class import AssetClass

_REQUIRED_COLUMNS = (
    "asset_class",
    "dist_metro_m",
    "count_stations_1km",
    "count_apartments_500m",
    "count_houses_500m",
    "count_commercial_500m",
    "road_length_500m",
)

_TARGET_COLUMN = "synthetic_target_rub_per_m2"

_BASE_BY_CLASS = {
    AssetClass.APARTMENT.value: 80_000.0,
    AssetClass.HOUSE.value: 50_000.0,
    AssetClass.COMMERCIAL.value: 100_000.0,
}

_NOISE_SIGMA_BY_CLASS = {
    AssetClass.APARTMENT.value: 10_000.0,
    AssetClass.HOUSE.value: 8_000.0,
    AssetClass.COMMERCIAL.value: 15_000.0,
}


def _validate_inputs(df: pl.DataFrame) -> None:
    for col in _REQUIRED_COLUMNS:
        if col not in df.columns:
            raise KeyError(f"compute_object_synthetic_target requires column: {col}")

    classes = set(df["asset_class"].unique().to_list())
    unknown = classes - set(_BASE_BY_CLASS.keys())
    if unknown:
        raise ValueError(
            f"compute_object_synthetic_target got unknown asset_class values: {sorted(unknown)}"
        )


def _apartment_factor(df: pl.DataFrame) -> pl.Expr:
    metro = 1.0 + 0.6 * (-pl.col("dist_metro_m") / 800.0).exp()
    density = 1.0 + 0.05 * (
        pl.col("count_apartments_500m").cast(pl.Float64)
        + pl.col("count_commercial_500m").cast(pl.Float64)
    ).log1p()
    road = 1.0 + 0.0001 * pl.col("road_length_500m")
    _ = df  # Polars exprs reference column names, df not needed here
    return metro * density * road


def _house_factor(df: pl.DataFrame) -> pl.Expr:
    proximity = 1.0 + 0.3 * (-pl.col("dist_metro_m") / 5000.0).exp()
    cluster = 1.0 + 0.04 * pl.col("count_houses_500m").cast(pl.Float64).log1p()
    crowding = (1.0 - 0.02 * pl.col("count_apartments_500m").cast(pl.Float64).log1p()).clip(
        lower_bound=0.6, upper_bound=1.0
    )
    road = 1.0 + 0.00005 * pl.col("road_length_500m")
    _ = df
    return proximity * cluster * crowding * road


def _commercial_factor(df: pl.DataFrame) -> pl.Expr:
    metro = 1.0 + 1.0 * (-pl.col("dist_metro_m") / 1500.0).exp()
    foot = 1.0 + 0.1 * (
        pl.col("count_apartments_500m").cast(pl.Float64)
        + pl.col("count_commercial_500m").cast(pl.Float64)
        + pl.col("count_houses_500m").cast(pl.Float64)
    ).log1p()
    road = 1.0 + 0.0002 * pl.col("road_length_500m")
    _ = df
    return metro * foot * road


def compute_object_synthetic_target(df: pl.DataFrame, *, seed: int) -> pl.DataFrame:
    _validate_inputs(df)

    base = pl.col("asset_class").replace_strict(_BASE_BY_CLASS)
    sigma = pl.col("asset_class").replace_strict(_NOISE_SIGMA_BY_CLASS)

    factor = (
        pl.when(pl.col("asset_class") == AssetClass.APARTMENT.value)
        .then(_apartment_factor(df))
        .when(pl.col("asset_class") == AssetClass.HOUSE.value)
        .then(_house_factor(df))
        .when(pl.col("asset_class") == AssetClass.COMMERCIAL.value)
        .then(_commercial_factor(df))
        .otherwise(1.0)
    )

    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(df.height)

    return df.with_columns(
        ((base * factor) + sigma * pl.Series("_noise", noise))
        .clip(lower_bound=0.0)
        .alias(_TARGET_COLUMN)
    )
