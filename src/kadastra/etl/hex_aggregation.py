"""Per-hex aggregation of per-object features and OOF predictions.

Produces one row per (resolution, h3_index, asset_class) where
``asset_class == "all"`` is the cross-class roll-up. The output feeds
the map UI's hex-mode (median target / median OOF prediction / object
counts at any of res=7/8/9/10) and the per-hex inspector.

Inputs:

- ``objects``: per-object DataFrame with at least ``(object_id,
  asset_class, lat, lon, synthetic_target_rub_per_m2)``. Optionally
  carries ``y_pred_oof`` and ``fold_id`` after a left-join with the
  OOF predictions parquet (``y_pred_oof`` may be all-null when the
  per-class OOF artifact is missing).
- ``resolution``: H3 resolution to aggregate at.

Output schema:

- ``h3_index``    (Utf8)
- ``resolution``  (Int32)
- ``asset_class`` (Utf8) — class value or ``"all"``
- ``count``       (Int64) — number of objects in the cell
- ``median_target_rub_per_m2`` (Float64)
- ``median_pred_oof_rub_per_m2`` (Float64 / nullable)
- ``median_residual_rub_per_m2`` (Float64 / nullable, = pred − true)
- numeric means: ``mean_levels``, ``mean_flats``, ``mean_area_m2``,
  ``mean_year_built`` — only those present in ``objects``.
- categorical mode: ``dominant_intra_city_raion``,
  ``dominant_mun_okrug_name``, ``dominant_settlement_name`` — only
  those present in ``objects``.
"""

from __future__ import annotations

from collections.abc import Iterable

import h3
import polars as pl

_TARGET = "synthetic_target_rub_per_m2"
_PRED = "y_pred_oof"
# Per-hex means. The aggregator only emits a `mean_<col>` row for
# columns actually present in the input — so dropping an OSM extract
# (e.g., no landfill polygons) just drops the matching map option,
# never an all-null column. The list below is the full curated set
# the map UI knows how to display; presence per-region varies.
_NUMERIC_MEANS: tuple[str, ...] = (
    # Building / land descriptors
    "levels", "flats", "area_m2", "year_built", "age_years",
    # ADR-0019 distance to nearest geometry of each layer (polygonal,
    # linear, point-POI). Names follow the ``dist_to_<layer>_m``
    # convention used by build_object_features for ADR-0019 layers;
    # the legacy ``dist_metro_m`` / ``dist_entrance_m`` predate that
    # convention and stay as-is.
    "dist_to_water_m", "dist_to_park_m", "dist_to_forest_m",
    "dist_to_industrial_m",
    "dist_to_cemetery_m", "dist_to_landfill_m",
    "dist_to_powerline_m", "dist_to_railway_m",
    "dist_to_school_m", "dist_to_kindergarten_m", "dist_to_clinic_m",
    "dist_to_hospital_m", "dist_to_pharmacy_m", "dist_to_supermarket_m",
    "dist_to_cafe_m", "dist_to_restaurant_m",
    "dist_to_bus_stop_m", "dist_to_tram_stop_m", "dist_to_railway_station_m",
    "dist_metro_m", "dist_entrance_m",
    # Polygonal share-in-buffer at the canonical 500 m radius. The
    # other radii (100/300/800 m) are also on the per-object frame
    # but only one is exposed on the hex map to keep the feature
    # picker readable.
    "water_share_500m", "park_share_500m", "forest_share_500m",
    "industrial_share_500m", "cemetery_share_500m",
    # Road density.
    "road_length_500m",
    # Zonal density. Pre-existing ``count_*`` columns (legacy naming)
    # plus per-POI ``<layer>_within_500m`` columns produced by the
    # ADR-0019 zonal pipeline.
    "count_stations_1km", "count_entrances_500m",
    "count_apartments_500m", "count_houses_500m", "count_commercial_500m",
    "school_within_500m", "kindergarten_within_500m",
    "clinic_within_500m", "hospital_within_500m",
    "pharmacy_within_500m", "supermarket_within_500m",
    "cafe_within_500m", "restaurant_within_500m",
    "bus_stop_within_500m", "tram_stop_within_500m",
    "railway_station_within_500m",
)
_CATEGORICAL_MODES: tuple[str, ...] = (
    "intra_city_raion",
    "mun_okrug_name",
    "settlement_name",
)


def aggregate_objects_to_hex(
    objects: pl.DataFrame, *, resolution: int
) -> pl.DataFrame:
    if objects.is_empty():
        return _empty_output()

    h3_indices = [
        h3.latlng_to_cell(float(lat), float(lon), resolution)
        for lat, lon in zip(
            objects["lat"].to_list(), objects["lon"].to_list(), strict=True
        )
    ]
    enriched = objects.with_columns(pl.Series("h3_index", h3_indices, dtype=pl.Utf8))

    if _PRED not in enriched.columns:
        enriched = enriched.with_columns(
            pl.lit(None, dtype=pl.Float64).alias(_PRED)
        )

    enriched = enriched.with_columns(
        (pl.col(_PRED) - pl.col(_TARGET)).alias("_residual")
    )

    numeric_means = [c for c in _NUMERIC_MEANS if c in enriched.columns]
    categorical_modes = [c for c in _CATEGORICAL_MODES if c in enriched.columns]

    per_class = _aggregate(
        enriched, group_keys=("h3_index", "asset_class"),
        numeric_means=numeric_means, categorical_modes=categorical_modes,
    )
    cross_class = _aggregate(
        enriched.with_columns(pl.lit("all", dtype=pl.Utf8).alias("asset_class")),
        group_keys=("h3_index", "asset_class"),
        numeric_means=numeric_means, categorical_modes=categorical_modes,
    )

    out = pl.concat([per_class, cross_class], how="vertical_relaxed")
    return out.with_columns(
        pl.lit(resolution, dtype=pl.Int32).alias("resolution")
    ).sort(["resolution", "h3_index", "asset_class"])


def _empty_output() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "h3_index": pl.Utf8,
            "resolution": pl.Int32,
            "asset_class": pl.Utf8,
            "count": pl.Int64,
            "median_target_rub_per_m2": pl.Float64,
            "median_pred_oof_rub_per_m2": pl.Float64,
            "median_residual_rub_per_m2": pl.Float64,
        }
    )


def _aggregate(
    df: pl.DataFrame,
    *,
    group_keys: tuple[str, ...],
    numeric_means: Iterable[str],
    categorical_modes: Iterable[str],
) -> pl.DataFrame:
    aggs = [
        pl.len().alias("count"),
        pl.col(_TARGET).median().alias("median_target_rub_per_m2"),
        pl.col(_PRED).median().alias("median_pred_oof_rub_per_m2"),
        pl.col("_residual").median().alias("median_residual_rub_per_m2"),
    ]
    for col in numeric_means:
        aggs.append(pl.col(col).cast(pl.Float64).mean().alias(f"mean_{col}"))
    for col in categorical_modes:
        # Mode (most-frequent value). polars has ``mode()`` returning a
        # Series of all most-frequent values; we pick the first.
        aggs.append(pl.col(col).drop_nulls().mode().first().alias(f"dominant_{col}"))
    return df.group_by(list(group_keys), maintain_order=False).agg(aggs)
