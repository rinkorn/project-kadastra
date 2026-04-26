"""Tests for select_object_feature_columns.

Both TrainObjectValuationModel and InferObjectValuation share the
same rule for choosing model-feature columns from a per-object
DataFrame. The shared helper has to:

- exclude identifier and target columns (object_id, lat/lon, target),
- exclude the relative-feature *bookkeeping* columns
  (``parent_h3_p{R}``) since they are high-cardinality IDs that
  vanish under spatial CV (a whole parent is held out, so the IDs
  in the test fold are unknown to the model — they add noise),
- keep the *aggregate* and *derivative* columns introduced by
  ADR-0012 (``count_p{R}``, ``F__rel_p{R}_*``) since those are
  proper numeric features,
- preserve column order so ``cat_feature_indices`` lines up with
  numeric-then-categorical layout.
"""

from __future__ import annotations

from collections.abc import Mapping

import polars as pl

from kadastra.ml.object_feature_columns import select_object_feature_columns


def _df(schema: Mapping[str, pl.DataType | type[pl.DataType]]) -> pl.DataFrame:
    return pl.DataFrame(schema=dict(schema))


def test_excludes_identifier_and_target_columns() -> None:
    df = _df(
        {
            "object_id": pl.Utf8,
            "asset_class": pl.Utf8,
            "lat": pl.Float64,
            "lon": pl.Float64,
            "synthetic_target_rub_per_m2": pl.Float64,
            "cost_value_rub": pl.Float64,
            "levels": pl.Int64,
        }
    )
    numeric, categorical = select_object_feature_columns(df)
    assert numeric == ["levels"]
    assert categorical == []


def test_excludes_parent_h3_p_id_columns() -> None:
    df = _df(
        {
            "object_id": pl.Utf8,
            "asset_class": pl.Utf8,
            "lat": pl.Float64,
            "lon": pl.Float64,
            "synthetic_target_rub_per_m2": pl.Float64,
            "cost_value_rub": pl.Float64,
            "levels": pl.Int64,
            "materials": pl.Utf8,
            "parent_h3_p7": pl.Utf8,
            "parent_h3_p8": pl.Utf8,
            "count_p7": pl.UInt32,
            "count_p8": pl.UInt32,
        }
    )
    numeric, categorical = select_object_feature_columns(df)
    assert "parent_h3_p7" not in categorical
    assert "parent_h3_p7" not in numeric
    assert "parent_h3_p8" not in categorical
    assert "parent_h3_p8" not in numeric
    # count_p{R} is a normal numeric feature — keep it.
    assert "count_p7" in numeric
    assert "count_p8" in numeric
    # materials remains a regular categorical.
    assert "materials" in categorical
    assert "levels" in numeric


def test_keeps_relative_derivative_columns() -> None:
    df = _df(
        {
            "object_id": pl.Utf8,
            "asset_class": pl.Utf8,
            "lat": pl.Float64,
            "lon": pl.Float64,
            "synthetic_target_rub_per_m2": pl.Float64,
            "cost_value_rub": pl.Float64,
            "dist_metro_m": pl.Float64,
            "dist_metro_m__rel_p7_diff_med": pl.Float64,
            "dist_metro_m__rel_p7_ratio_med": pl.Float64,
            "dist_metro_m__rel_p7_z_iqr": pl.Float64,
            "parent_h3_p7": pl.Utf8,
        }
    )
    numeric, _ = select_object_feature_columns(df)
    for col in (
        "dist_metro_m",
        "dist_metro_m__rel_p7_diff_med",
        "dist_metro_m__rel_p7_ratio_med",
        "dist_metro_m__rel_p7_z_iqr",
    ):
        assert col in numeric


def test_excludes_polygon_wkt_3857_passthrough_column() -> None:
    """ADR-0017 prokids ``polygon_wkt_3857`` through gold for the
    inspector — it is identity/provenance, not a feature. With ~290k
    unique WKT strings it would otherwise become a high-cardinality
    categorical that collapses CatBoost cat-handling on val folds and
    breaks EBM bin-encoding outright."""
    df = _df(
        {
            "object_id": pl.Utf8,
            "polygon_wkt_3857": pl.Utf8,
            "levels": pl.Int64,
            "materials": pl.Utf8,
        }
    )
    numeric, categorical = select_object_feature_columns(df)
    assert "polygon_wkt_3857" not in categorical
    assert "polygon_wkt_3857" not in numeric
    # Other Utf8 columns still flow into categorical normally.
    assert "materials" in categorical
    # Other numeric columns still flow into numeric normally.
    assert "levels" in numeric


def test_preserves_column_order_within_groups() -> None:
    df = _df(
        {
            "object_id": pl.Utf8,
            "lat": pl.Float64,
            "lon": pl.Float64,
            "synthetic_target_rub_per_m2": pl.Float64,
            "cost_value_rub": pl.Float64,
            "asset_class": pl.Utf8,
            "levels": pl.Int64,
            "year_built": pl.Int64,
            "area_m2": pl.Float64,
            "materials": pl.Utf8,
            "parent_h3_p7": pl.Utf8,
        }
    )
    numeric, categorical = select_object_feature_columns(df)
    # Numeric columns appear in the same order as in the source frame.
    assert numeric == ["levels", "year_built", "area_m2"]
    assert categorical == ["materials"]
