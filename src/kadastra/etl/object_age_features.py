"""ADR-0020: derive age and era features from ``year_built``.

Pure function over a polars DataFrame — no ports, no I/O.
``current_year`` is passed explicitly so the same data + same year
always produce the same output (no system-clock dependency).
"""

from __future__ import annotations

import polars as pl

# (lo, hi, code) — closed interval [lo, hi], inclusive on both ends.
# `lo=None` means "open at the lower end" (everything below `hi`).
# `hi=None` means "open at the upper end" (everything from `lo`).
# Order doesn't matter — bands are non-overlapping by construction.
_ERA_BINS: list[tuple[int | None, int | None, str]] = [
    (None, 1916, "pre_revolution"),
    (1917, 1945, "early_soviet"),
    (1946, 1956, "stalin"),
    (1957, 1968, "khrushchev"),
    (1969, 1980, "brezhnev"),
    (1981, 1991, "late_soviet"),
    (1992, 2000, "90s_transition"),
    (2001, 2010, "2000s"),
    (2011, 2020, "2010s"),
    (2021, None, "new_2020+"),
]


def compute_object_age_features(objects: pl.DataFrame, *, current_year: int) -> pl.DataFrame:
    """Append ``age_years``, ``age_years_sq``, ``era_category``,
    ``is_new_construction`` columns derived from ``year_built``.

    Edge cases (data-quality):
    - ``year_built`` is ``null`` → all four features null/"unknown".
    - ``year_built`` is ``0`` (NSPD's «год не указан» encoding) →
      treated identically to null.
    """
    if "year_built" not in objects.columns:
        raise KeyError("compute_object_age_features requires column 'year_built'")

    # Treat year_built == 0 as null (data-quality from NSPD).
    yb = pl.when(pl.col("year_built") == 0).then(None).otherwise(pl.col("year_built"))

    age = (current_year - yb).cast(pl.Int64)

    era_expr = pl.lit("unknown", dtype=pl.Utf8)
    for lo, hi, code in _ERA_BINS:
        cond = pl.lit(True)
        if lo is not None:
            cond = cond & (yb >= lo)
        if hi is not None:
            cond = cond & (yb <= hi)
        era_expr = pl.when(yb.is_not_null() & cond).then(pl.lit(code, dtype=pl.Utf8)).otherwise(era_expr)

    return objects.with_columns(
        age.alias("age_years"),
        (age * age).alias("age_years_sq"),
        era_expr.alias("era_category"),
        (age <= 5).alias("is_new_construction"),
    )
