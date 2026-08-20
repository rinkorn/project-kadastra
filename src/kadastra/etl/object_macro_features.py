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
    raise NotImplementedError
