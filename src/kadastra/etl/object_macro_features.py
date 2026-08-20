"""Block 4b — macro-territorial features per object (ADR-0022).

Attaches up to 6 EMISS-derived columns to a per-object DataFrame by
left-joining the wide ``silver/macro_oktmo_features`` table on the
8-digit municipal OKTMO (``oktmo_full`` prefix):

- ``oktmo_avg_salary_rub``             (Float64) — avg monthly salary.
- ``oktmo_population``                 (Float64) — resident population.
- ``oktmo_population_density``         (Float64) — pop / area km².
- ``oktmo_housing_volume_5y_m2``       (Float64) — housing commissioned, 5y sum.
- ``oktmo_unemployment_pct``           (Float64) — unemployment rate.
- ``oktmo_retail_turnover_per_capita`` (Float64) — retail turnover / pop.

Objects without a matching OKTMO (or without ``oktmo_full`` at all)
get null features — a no-match is a legitimate state, not an error.
"""

from __future__ import annotations

import polars as pl

MACRO_FEATURE_COLUMNS: tuple[str, ...] = (
    "oktmo_avg_salary_rub",
    "oktmo_population",
    "oktmo_population_density",
    "oktmo_housing_volume_5y_m2",
    "oktmo_unemployment_pct",
    "oktmo_retail_turnover_per_capita",
)


def compute_object_macro_features(
    objects: pl.DataFrame,
    *,
    macro_table: pl.DataFrame,
    target_year: int,
) -> pl.DataFrame:
    """Left-join macro-territorial features onto objects.

    ``macro_table`` is the wide per-(oktmo, year) table from
    ``silver/macro_oktmo_features`` (may carry several year partitions).
    The join key is the 8-digit municipal OKTMO: EMISS publishes at
    municipal grain, while GAR-derived ``oktmo_full`` is at settlement
    grain (11 digits), so objects join on ``oktmo_full[:8]``.

    Year alignment (ADR-0022 §Year alignment): each feature
    independently takes its last non-null value with
    ``year <= target_year`` — EMISS indicators publish with different
    lags, so a single per-row year cut would drop fresher features.

    Idempotent: pre-existing output columns are dropped first (the
    store is read-write; a rerun reads its own enriched output).
    """
    drop_existing = [c for c in MACRO_FEATURE_COLUMNS if c in objects.columns]
    if drop_existing:
        objects = objects.drop(drop_existing)

    if objects.height == 0 or "oktmo_full" not in objects.columns:
        return objects.with_columns([pl.lit(None, dtype=pl.Float64).alias(c) for c in MACRO_FEATURE_COLUMNS])

    objects = objects.with_columns(
        pl.col("oktmo_full").str.slice(0, 8).alias("_oktmo8"),
    )

    enriched = objects
    for col in MACRO_FEATURE_COLUMNS:
        if col not in macro_table.columns:
            enriched = enriched.with_columns(pl.lit(None, dtype=pl.Float64).alias(col))
            continue
        # Last available year <= target_year per oktmo for THIS feature.
        latest = (
            macro_table.lazy()
            .filter(pl.col("year") <= target_year, pl.col(col).is_not_null())
            .group_by("oktmo")
            .agg(pl.col(col).sort_by("year").last())
            .collect()
            .rename({"oktmo": "_oktmo8"})
        )
        enriched = enriched.join(latest, on="_oktmo8", how="left")

    return enriched.drop("_oktmo8")
