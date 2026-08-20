"""Tests for join_road_class_features (ADR-0024, group 1).

The BuildObjectFeatures pipeline consumes the silver
``road_class_per_object`` table via a plain LEFT JOIN on ``object_id``;
these tests pin the join contract: expected columns, null fallback for
objects missing from the table, and idempotency on reruns.
"""

import polars as pl

from kadastra.etl.object_road_class_features import (
    ROAD_CLASS_FEATURE_COLUMNS,
    join_road_class_features,
)


def _objects() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {"object_id": "x", "asset_class": "house", "lat": 55.78, "lon": 49.12},
            {"object_id": "y", "asset_class": "house", "lat": 55.79, "lon": 49.13},
        ]
    )


def _road_features() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "object_id": "x",
                "nearest_road_class": "residential",
                "dist_to_motorway_m": 1500.0,
                "dist_to_primary_m": 800.0,
                "dist_to_secondary_m": 300.0,
                "dist_to_residential_m": 12.5,
                "dist_to_pedestrian_m": 40.0,
            }
        ],
        schema={
            "object_id": pl.Utf8,
            "nearest_road_class": pl.Utf8,
            "dist_to_motorway_m": pl.Float64,
            "dist_to_primary_m": pl.Float64,
            "dist_to_secondary_m": pl.Float64,
            "dist_to_residential_m": pl.Float64,
            "dist_to_pedestrian_m": pl.Float64,
        },
    )


def test_join_produces_expected_columns() -> None:
    out = join_road_class_features(_objects(), _road_features())

    for col in ROAD_CLASS_FEATURE_COLUMNS:
        assert col in out.columns
    row_x = out.filter(pl.col("object_id") == "x").row(0, named=True)
    assert row_x["nearest_road_class"] == "residential"
    assert row_x["dist_to_residential_m"] == 12.5


def test_objects_missing_from_table_get_nulls() -> None:
    out = join_road_class_features(_objects(), _road_features())

    row_y = out.filter(pl.col("object_id") == "y").row(0, named=True)
    assert row_y["nearest_road_class"] is None
    assert row_y["dist_to_motorway_m"] is None


def test_empty_table_yields_null_columns() -> None:
    empty = _road_features().head(0)
    out = join_road_class_features(_objects(), empty)

    assert out.height == 2
    for col in ROAD_CLASS_FEATURE_COLUMNS:
        assert out[col].null_count() == 2


def test_join_is_idempotent() -> None:
    once = join_road_class_features(_objects(), _road_features())
    # Rerun reads its own enriched output: pre-existing columns must be
    # replaced, not duplicated as *_right.
    twice = join_road_class_features(once, _road_features())

    assert not any(c.endswith("_right") for c in twice.columns)
    assert twice.columns == once.columns
    row_x = twice.filter(pl.col("object_id") == "x").row(0, named=True)
    assert row_x["nearest_road_class"] == "residential"
