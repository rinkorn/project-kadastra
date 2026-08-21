"""Tests for cell reference-object templates (ADR-0029, product 1)."""

from __future__ import annotations

import polars as pl
import pytest

from kadastra.etl.cell_reference_object import (
    ReferenceObject,
    build_reference_objects,
)


def _objects_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "levels": [2, 2, 3, None],
            "flats": [0, 0, 1, 0],
            "area_m2": [60.0, 80.0, 100.0, None],
            "year_built": [1970, 1980, 1990, 0],
            "underground_floors": [0, 1, 0, None],
            "polygon_area_m2": [600.0, 800.0, 1000.0, None],
            "materials": ["brick", "panel", "brick", None],
            "vri": ["ИЖС", "ИЖС", "Садоводство", None],
            "category_zem": ["ЗНП", "ЗНП", "СНТ", None],
        }
    )


def test_default_variant_uses_median_for_numeric_and_mode_for_categorical() -> None:
    refs = build_reference_objects(_objects_frame(), current_year=2026)
    assert len(refs) == 1
    ref = refs[0]
    assert ref.variant == "default"
    attrs = ref.attributes
    # Medians (nulls ignored); int-typed attrs rounded to int.
    assert attrs["levels"] == 2
    assert attrs["area_m2"] == 80.0
    assert attrs["underground_floors"] == 0
    assert attrs["polygon_area_m2"] == 800.0
    assert attrs["materials"] == "brick"
    assert attrs["vri"] == "ИЖС"
    assert attrs["category_zem"] == "ЗНП"


def test_year_built_zero_counts_as_missing() -> None:
    refs = build_reference_objects(_objects_frame(), current_year=2026)
    # Median of [1970, 1980, 1990] — the 0 is excluded (NSPD «год не указан»).
    assert refs[0].attributes["year_built"] == 1980


def test_derived_age_features_recomputed_from_median_year() -> None:
    refs = build_reference_objects(_objects_frame(), current_year=2026)
    attrs = refs[0].attributes
    assert attrs["age_years"] == 2026 - 1980
    assert attrs["age_years_sq"] == (2026 - 1980) ** 2
    assert attrs["era_category"] == "brezhnev"


def test_vri_variants_built_for_top_n() -> None:
    refs = build_reference_objects(_objects_frame(), current_year=2026, vri_top_n=2)
    variants = [r.variant for r in refs]
    assert variants[0] == "default"
    assert set(variants[1:]) == {"ИЖС", "Садоводство"}
    izhs = next(r for r in refs if r.variant == "ИЖС")
    assert izhs.attributes["vri"] == "ИЖС"
    # Medians within the ИЖС slice only.
    assert izhs.attributes["area_m2"] == 70.0


def test_all_null_column_yields_null_attribute() -> None:
    df = _objects_frame().with_columns(pl.lit(None, dtype=pl.Float64).alias("area_m2")).drop("area_m2")
    df = df.rename({"area_m2": "area_m2"}) if "area_m2" in df.columns else df
    df = pl.DataFrame({**{c: _objects_frame()[c] for c in _objects_frame().columns if c != "area_m2"}})
    df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("area_m2"))
    refs = build_reference_objects(df, current_year=2026)
    assert refs[0].attributes["area_m2"] is None
    # year_built still present → derived age fields unaffected.
    assert refs[0].attributes["age_years"] == 46


def test_empty_frame_raises() -> None:
    empty = _objects_frame().head(0)
    with pytest.raises(ValueError, match="empty"):
        build_reference_objects(empty, current_year=2026)


def test_reference_object_is_frozen() -> None:
    refs = build_reference_objects(_objects_frame(), current_year=2026)
    assert isinstance(refs[0], ReferenceObject)
    with pytest.raises(AttributeError):
        refs[0].variant = "mutated"  # type: ignore[misc]
