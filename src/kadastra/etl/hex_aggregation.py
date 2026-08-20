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
- numeric means of per-object descriptors: ``mean_levels``,
  ``mean_flats``, ``mean_area_m2``, ``mean_year_built``,
  ``mean_age_years`` — only those present in ``objects``. Слой-1 ЦОФ
  are intentionally NOT mean-reduced here (the map reads them from
  the cell ЦОФ store via ``/api/cell_tsorf``).
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
# columns actually present in the input — so a missing source column
# just drops the matching map option, never an all-null column.
#
# Deliberately limited to per-object DESCRIPTORS (building/land
# attributes). Слой-1 ЦОФ (dist_to_*, *_share_*, *_within_*,
# count_*, road_length, metro) are NOT duplicated here: the map reads
# those from the cell ЦОФ store via /api/cell_tsorf (ADR-0027), which
# covers the full grid, not just cells with objects.
_NUMERIC_MEANS: tuple[str, ...] = (
    "levels",
    "flats",
    "area_m2",
    "year_built",
    "age_years",
)
_CATEGORICAL_MODES: tuple[str, ...] = (
    "intra_city_raion",
    "mun_okrug_name",
    "settlement_name",
)


def aggregate_objects_to_hex(objects: pl.DataFrame, *, resolution: int) -> pl.DataFrame:
    if objects.is_empty():
        return _empty_output()

    h3_indices = [
        h3.latlng_to_cell(float(lat), float(lon), resolution)
        for lat, lon in zip(objects["lat"].to_list(), objects["lon"].to_list(), strict=True)
    ]
    enriched = objects.with_columns(pl.Series("h3_index", h3_indices, dtype=pl.Utf8))

    if _PRED not in enriched.columns:
        enriched = enriched.with_columns(pl.lit(None, dtype=pl.Float64).alias(_PRED))

    enriched = enriched.with_columns((pl.col(_PRED) - pl.col(_TARGET)).alias("_residual"))

    numeric_means = [c for c in _NUMERIC_MEANS if c in enriched.columns]
    categorical_modes = [c for c in _CATEGORICAL_MODES if c in enriched.columns]

    per_class = _aggregate(
        enriched,
        group_keys=("h3_index", "asset_class"),
        numeric_means=numeric_means,
        categorical_modes=categorical_modes,
    )
    cross_class = _aggregate(
        enriched.with_columns(pl.lit("all", dtype=pl.Utf8).alias("asset_class")),
        group_keys=("h3_index", "asset_class"),
        numeric_means=numeric_means,
        categorical_modes=categorical_modes,
    )

    out = pl.concat([per_class, cross_class], how="vertical_relaxed")
    return out.with_columns(pl.lit(resolution, dtype=pl.Int32).alias("resolution")).sort(
        ["resolution", "h3_index", "asset_class"]
    )


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
