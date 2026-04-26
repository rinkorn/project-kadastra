"""Unit tests for ADR-0020 — derive age + era features from year_built.

The computation is fully deterministic w.r.t. ``current_year`` (passed
explicitly so the same data + same year always produces the same
output, no system-clock dependency)."""

from __future__ import annotations

import polars as pl
import pytest

from kadastra.etl.object_age_features import compute_object_age_features


def _objects(years: list[int | None]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "object_id": [f"way/{i}" for i in range(len(years))],
            "year_built": pl.Series(years, dtype=pl.Int64),
        }
    )


def test_age_features_basic_2020_object() -> None:
    df = compute_object_age_features(_objects([2020]), current_year=2026)
    row = df.row(0, named=True)
    assert row["age_years"] == 6
    assert row["age_years_sq"] == 36
    assert row["era_category"] == "2010s"
    assert row["is_new_construction"] is False


def test_new_construction_flag_is_true_for_age_le_5() -> None:
    df = compute_object_age_features(
        _objects([2024, 2021, 2020, 2019]), current_year=2026
    )
    flags = df["is_new_construction"].to_list()
    assert flags == [True, True, True, False]


def test_new_construction_flag_is_true_for_current_year_build() -> None:
    df = compute_object_age_features(_objects([2026]), current_year=2026)
    row = df.row(0, named=True)
    assert row["age_years"] == 0
    assert row["era_category"] == "new_2020+"
    assert row["is_new_construction"] is True


def test_age_years_sq_is_squared_age() -> None:
    df = compute_object_age_features(
        _objects([2020, 2010, 1990]), current_year=2026
    )
    ages = df["age_years"].to_list()
    sqs = df["age_years_sq"].to_list()
    assert sqs == [a * a for a in ages]


def test_era_boundaries_match_adr_table() -> None:
    """Boundaries from ADR-0020 §«Эпохи»."""
    boundary_years_to_era = {
        1900: "pre_revolution",
        1916: "pre_revolution",
        1917: "early_soviet",
        1945: "early_soviet",
        1946: "stalin",
        1956: "stalin",
        1957: "khrushchev",
        1968: "khrushchev",
        1969: "brezhnev",
        1980: "brezhnev",
        1981: "late_soviet",
        1991: "late_soviet",
        1992: "90s_transition",
        2000: "90s_transition",
        2001: "2000s",
        2010: "2000s",
        2011: "2010s",
        2020: "2010s",
        2021: "new_2020+",
        2026: "new_2020+",
    }
    years = list(boundary_years_to_era.keys())
    df = compute_object_age_features(_objects(years), current_year=2026)
    eras = df["era_category"].to_list()
    expected = list(boundary_years_to_era.values())
    assert eras == expected, (
        f"era binning mismatch:\n"
        f"  years    : {years}\n"
        f"  got      : {eras}\n"
        f"  expected : {expected}"
    )


def test_null_year_built_yields_unknown_era_and_null_numerics() -> None:
    df = compute_object_age_features(_objects([None]), current_year=2026)
    row = df.row(0, named=True)
    assert row["age_years"] is None
    assert row["age_years_sq"] is None
    assert row["era_category"] == "unknown"
    assert row["is_new_construction"] is None


def test_zero_year_built_treated_as_unknown() -> None:
    """Data-quality edge case — NSPD occasionally encodes
    «год не указан» as 0 instead of null. Treat both the same."""
    df = compute_object_age_features(_objects([0]), current_year=2026)
    row = df.row(0, named=True)
    assert row["age_years"] is None
    assert row["age_years_sq"] is None
    assert row["era_category"] == "unknown"
    assert row["is_new_construction"] is None


def test_preserves_other_columns() -> None:
    df_in = pl.DataFrame(
        {
            "object_id": ["way/1", "way/2"],
            "year_built": pl.Series([2020, 1965], dtype=pl.Int64),
            "lat": [55.6, 55.7],
            "lon": [49.0, 49.1],
        }
    )
    df_out = compute_object_age_features(df_in, current_year=2026)
    assert df_out["object_id"].to_list() == ["way/1", "way/2"]
    assert df_out["lat"].to_list() == [55.6, 55.7]
    assert df_out["lon"].to_list() == [49.0, 49.1]
    # Original year_built also preserved.
    assert df_out["year_built"].to_list() == [2020, 1965]


def test_missing_year_built_column_raises() -> None:
    df = pl.DataFrame({"object_id": ["way/1"], "lat": [55.6]})
    with pytest.raises(KeyError, match="year_built"):
        compute_object_age_features(df, current_year=2026)
