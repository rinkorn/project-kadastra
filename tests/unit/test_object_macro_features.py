"""Unit tests for ADR-0022 — macro-territorial EMISS features per object.

The join is against a wide per-(oktmo, year) macro table built by
``scripts/build_macro_oktmo_features.py``. Key normalization: EMISS
publishes municipal-level OKTMO (8 digits); objects carry GAR-derived
``oktmo_full`` at settlement grain (8 or 11 digits), so the join key
is the 8-digit prefix. Each feature independently takes its
last-available value with ``year <= target_year``."""

from __future__ import annotations

import polars as pl

from kadastra.etl.object_macro_features import (
    MACRO_FEATURE_COLUMNS,
    compute_object_macro_features,
)


def _objects(oktmo_full: list[str | None]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "object_id": [f"way/{i}" for i in range(len(oktmo_full))],
            "oktmo_full": pl.Series(oktmo_full, dtype=pl.Utf8),
        }
    )


def _macro(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "oktmo": pl.Utf8,
            "year": pl.Int64,
            "oktmo_avg_salary_rub": pl.Float64,
            "oktmo_population": pl.Float64,
            "oktmo_population_density": pl.Float64,
            "oktmo_housing_volume_5y_m2": pl.Float64,
            "oktmo_unemployment_pct": pl.Float64,
            "oktmo_retail_turnover_per_capita": pl.Float64,
        },
    )


def test_joins_features_by_8digit_oktmo_prefix() -> None:
    """11-digit settlement OKTMO joins on its 8-digit municipal prefix."""
    objects = _objects(["92633412101", "92601000"])
    macro = _macro(
        [
            {
                "oktmo": "92633412",
                "year": 2024,
                "oktmo_avg_salary_rub": 70000.0,
                "oktmo_population": 5000.0,
                "oktmo_population_density": 50.0,
                "oktmo_housing_volume_5y_m2": 12000.0,
                "oktmo_unemployment_pct": 3.5,
                "oktmo_retail_turnover_per_capita": 200000.0,
            }
        ]
    )

    df = compute_object_macro_features(objects, macro_table=macro, target_year=2024)

    rows = df.sort("object_id").to_dicts()
    assert rows[0]["oktmo_avg_salary_rub"] == 70000.0
    assert rows[0]["oktmo_population"] == 5000.0
    assert rows[0]["oktmo_unemployment_pct"] == 3.5
    # Second object has no macro row → all features null.
    for col in MACRO_FEATURE_COLUMNS:
        assert rows[1][col] is None


def test_unknown_oktmo_yields_null_features() -> None:
    objects = _objects(["99999999"])
    macro = _macro(
        [
            {
                "oktmo": "92633412",
                "year": 2024,
                "oktmo_avg_salary_rub": 70000.0,
                "oktmo_population": 5000.0,
                "oktmo_population_density": 50.0,
                "oktmo_housing_volume_5y_m2": 12000.0,
                "oktmo_unemployment_pct": 3.5,
                "oktmo_retail_turnover_per_capita": 200000.0,
            }
        ]
    )

    df = compute_object_macro_features(objects, macro_table=macro, target_year=2024)

    row = df.row(0, named=True)
    for col in MACRO_FEATURE_COLUMNS:
        assert row[col] is None


def test_null_oktmo_yields_null_features() -> None:
    objects = _objects([None])
    macro = _macro(
        [
            {
                "oktmo": "92633412",
                "year": 2024,
                "oktmo_avg_salary_rub": 70000.0,
                "oktmo_population": 5000.0,
                "oktmo_population_density": 50.0,
                "oktmo_housing_volume_5y_m2": 12000.0,
                "oktmo_unemployment_pct": 3.5,
                "oktmo_retail_turnover_per_capita": 200000.0,
            }
        ]
    )

    df = compute_object_macro_features(objects, macro_table=macro, target_year=2024)

    row = df.row(0, named=True)
    for col in MACRO_FEATURE_COLUMNS:
        assert row[col] is None


def test_missing_oktmo_column_yields_null_features() -> None:
    """If the municipality block did not run (no ``oktmo_full`` at all),
    every row is a no-match — null columns, no crash."""
    objects = pl.DataFrame({"object_id": ["way/0"]})
    macro = _macro(
        [
            {
                "oktmo": "92633412",
                "year": 2024,
                "oktmo_avg_salary_rub": 70000.0,
                "oktmo_population": None,
                "oktmo_population_density": None,
                "oktmo_housing_volume_5y_m2": None,
                "oktmo_unemployment_pct": None,
                "oktmo_retail_turnover_per_capita": None,
            }
        ]
    )

    df = compute_object_macro_features(objects, macro_table=macro, target_year=2024)

    assert set(MACRO_FEATURE_COLUMNS).issubset(df.columns)
    assert df.row(0, named=True)["oktmo_avg_salary_rub"] is None


def test_year_alignment_takes_last_available_year() -> None:
    """target_year=2022 with data for 2020 and 2024 → 2020 wins."""
    objects = _objects(["92633412"])
    macro = _macro(
        [
            {
                "oktmo": "92633412",
                "year": 2020,
                "oktmo_avg_salary_rub": 50000.0,
                "oktmo_population": None,
                "oktmo_population_density": None,
                "oktmo_housing_volume_5y_m2": None,
                "oktmo_unemployment_pct": None,
                "oktmo_retail_turnover_per_capita": None,
            },
            {
                "oktmo": "92633412",
                "year": 2024,
                "oktmo_avg_salary_rub": 70000.0,
                "oktmo_population": None,
                "oktmo_population_density": None,
                "oktmo_housing_volume_5y_m2": None,
                "oktmo_unemployment_pct": None,
                "oktmo_retail_turnover_per_capita": None,
            },
        ]
    )

    df = compute_object_macro_features(objects, macro_table=macro, target_year=2022)

    assert df.row(0, named=True)["oktmo_avg_salary_rub"] == 50000.0


def test_year_alignment_is_per_feature() -> None:
    """Each feature takes its own last-available year (EMISS indicators
    publish with different lags)."""
    objects = _objects(["92633412"])
    macro = _macro(
        [
            {
                "oktmo": "92633412",
                "year": 2023,
                "oktmo_avg_salary_rub": 65000.0,
                "oktmo_population": 4900.0,
                "oktmo_population_density": None,
                "oktmo_housing_volume_5y_m2": None,
                "oktmo_unemployment_pct": None,
                "oktmo_retail_turnover_per_capita": None,
            },
            {
                "oktmo": "92633412",
                "year": 2024,
                "oktmo_avg_salary_rub": None,
                "oktmo_population": 5000.0,
                "oktmo_population_density": None,
                "oktmo_housing_volume_5y_m2": None,
                "oktmo_unemployment_pct": None,
                "oktmo_retail_turnover_per_capita": None,
            },
        ]
    )

    df = compute_object_macro_features(objects, macro_table=macro, target_year=2024)

    row = df.row(0, named=True)
    assert row["oktmo_avg_salary_rub"] == 65000.0  # 2023 — last non-null
    assert row["oktmo_population"] == 5000.0  # 2024


def test_idempotent_rerun_replaces_columns() -> None:
    """Re-running on an already-enriched frame replaces the macro columns
    instead of producing ``*_right`` duplicates (store is read-write)."""
    objects = _objects(["92633412"]).with_columns(
        pl.lit(1.0).alias("oktmo_avg_salary_rub"),
    )
    macro = _macro(
        [
            {
                "oktmo": "92633412",
                "year": 2024,
                "oktmo_avg_salary_rub": 70000.0,
                "oktmo_population": None,
                "oktmo_population_density": None,
                "oktmo_housing_volume_5y_m2": None,
                "oktmo_unemployment_pct": None,
                "oktmo_retail_turnover_per_capita": None,
            }
        ]
    )

    df = compute_object_macro_features(objects, macro_table=macro, target_year=2024)

    assert "oktmo_avg_salary_rub_right" not in df.columns
    assert df.row(0, named=True)["oktmo_avg_salary_rub"] == 70000.0


def test_empty_objects_frame_gets_null_columns() -> None:
    objects = pl.DataFrame(
        schema={"object_id": pl.Utf8, "oktmo_full": pl.Utf8},
    )
    macro = _macro(
        [
            {
                "oktmo": "92633412",
                "year": 2024,
                "oktmo_avg_salary_rub": 70000.0,
                "oktmo_population": None,
                "oktmo_population_density": None,
                "oktmo_housing_volume_5y_m2": None,
                "oktmo_unemployment_pct": None,
                "oktmo_retail_turnover_per_capita": None,
            }
        ]
    )

    df = compute_object_macro_features(objects, macro_table=macro, target_year=2024)

    assert df.height == 0
    assert set(MACRO_FEATURE_COLUMNS).issubset(df.columns)
    assert df.schema["oktmo_avg_salary_rub"] == pl.Float64
