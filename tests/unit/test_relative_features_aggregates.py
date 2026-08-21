"""Tests for applying training-time relative-feature aggregates (ADR-0029)."""

from __future__ import annotations

import h3
import polars as pl
import pytest

from kadastra.etl.relative_features import (
    compute_parent_aggregates,
    compute_relative_features,
    join_relative_features,
)

_LAT, _LON = 55.79, 49.11
_CELL = h3.latlng_to_cell(_LAT, _LON, 10)


def _objects_frame() -> pl.DataFrame:
    # Four objects in one res-7 parent area.
    return pl.DataFrame(
        {
            "lat": [_LAT, _LAT + 0.001, _LAT, _LAT + 0.001],
            "lon": [_LON, _LON, _LON + 0.001, _LON + 0.001],
            "dist_metro_m": [1000.0, 2000.0, 3000.0, None],
            "area_m2": [50.0, 60.0, 70.0, 80.0],
        }
    )


def test_aggregates_match_compute_relative_features_semantics() -> None:
    objects = _objects_frame()
    reference = compute_relative_features(objects, parent_resolutions=[7], feature_columns=["dist_metro_m"])

    aggregates = compute_parent_aggregates(objects, parent_resolutions=[7], feature_columns=["dist_metro_m"])
    applied = join_relative_features(objects, aggregates, parent_resolutions=[7], feature_columns=["dist_metro_m"])

    for col in (
        "count_p7",
        "dist_metro_m__rel_p7_diff_med",
        "dist_metro_m__rel_p7_ratio_med",
        "dist_metro_m__rel_p7_z_iqr",
    ):
        assert col in applied.columns
        got = applied[col].to_list()
        want = reference[col].to_list()
        for g, w in zip(got, want, strict=True):
            if w is None:
                assert g is None
            else:
                assert g == pytest.approx(w)


def test_new_rows_inherit_training_aggregates() -> None:
    objects = _objects_frame()
    aggregates = compute_parent_aggregates(objects, parent_resolutions=[7], feature_columns=["dist_metro_m"])

    # A new cell point in the same res-7 parent, not part of the training frame.
    new = pl.DataFrame({"lat": [_LAT + 0.0005], "lon": [_LON + 0.0005], "dist_metro_m": [2000.0]})
    applied = join_relative_features(new, aggregates, parent_resolutions=[7], feature_columns=["dist_metro_m"])
    # Median of [1000, 2000, 3000] = 2000 → diff 0, ratio 1.
    assert applied["dist_metro_m__rel_p7_diff_med"][0] == pytest.approx(0.0)
    assert applied["dist_metro_m__rel_p7_ratio_med"][0] == pytest.approx(1.0)
    assert applied["count_p7"][0] == 4


def test_row_in_unpopulated_parent_gets_nulls() -> None:
    objects = _objects_frame()
    aggregates = compute_parent_aggregates(objects, parent_resolutions=[7], feature_columns=["dist_metro_m"])

    far = pl.DataFrame({"lat": [55.2], "lon": [49.8], "dist_metro_m": [5000.0]})
    applied = join_relative_features(far, aggregates, parent_resolutions=[7], feature_columns=["dist_metro_m"])
    assert applied["dist_metro_m__rel_p7_diff_med"][0] is None
    assert applied["count_p7"][0] is None


def test_ratio_null_when_parent_median_zero() -> None:
    objects = _objects_frame().with_columns(pl.lit(0.0).alias("dist_metro_m"))
    aggregates = compute_parent_aggregates(objects, parent_resolutions=[7], feature_columns=["dist_metro_m"])
    applied = join_relative_features(objects, aggregates, parent_resolutions=[7], feature_columns=["dist_metro_m"])
    assert applied["dist_metro_m__rel_p7_ratio_med"][0] is None
    # diff vs median 0 is still defined.
    assert applied["dist_metro_m__rel_p7_diff_med"][0] == pytest.approx(0.0)
