"""Tests for compute_relative_features.

For each numeric ZOF column, the function adds derived features
relative to aggregates over the object's parent H3 cell at one or
more resolutions:

- ``F__rel_p{R}_diff_med``  = F − median(parent)
- ``F__rel_p{R}_ratio_med`` = F / median(parent)
- ``F__rel_p{R}_z_iqr``     = (F − median) / (p75 − p25)
- ``parent_h3_p{R}``        = the parent H3 index (helpful downstream)
- ``count_p{R}``            = how many objects share that parent

This is the methodological win described in info/grid-rationale.md
§11 ("главный выигрыш сетки"): a value is meaningful only against
its neighbourhood — see ADR-0012 for full rationale and contracts.
"""

from __future__ import annotations

import math

import h3
import numpy as np
import polars as pl

from kadastra.etl.relative_features import compute_relative_features


def _df(rows: list[dict[str, float]]) -> pl.DataFrame:
    schema = {
        "object_id": pl.Utf8,
        "lat": pl.Float64,
        "lon": pl.Float64,
        "dist_metro_m": pl.Float64,
    }
    return pl.DataFrame(rows, schema=schema)


# Coordinates inside Kazan, close enough that res=7 collapses them
# into a single parent hex (~1.4 km edge, ~5 km²) but spread enough
# to make medians/IQR non-trivial.
_KAZAN_CENTER = (55.7887, 49.1221)
_NEARBY = (55.7895, 49.1230)
_NEARBY2 = (55.7900, 49.1240)


def test_adds_derived_columns_per_feature_and_resolution() -> None:
    objects = _df(
        [
            {"object_id": "a", "lat": _KAZAN_CENTER[0], "lon": _KAZAN_CENTER[1], "dist_metro_m": 100.0},
            {"object_id": "b", "lat": _NEARBY[0], "lon": _NEARBY[1], "dist_metro_m": 200.0},
            {"object_id": "c", "lat": _NEARBY2[0], "lon": _NEARBY2[1], "dist_metro_m": 300.0},
        ]
    )

    out = compute_relative_features(
        objects, parent_resolutions=[7, 8], feature_columns=["dist_metro_m"]
    )

    expected_extra = {
        "parent_h3_p7", "count_p7",
        "parent_h3_p8", "count_p8",
        "dist_metro_m__rel_p7_diff_med",
        "dist_metro_m__rel_p7_ratio_med",
        "dist_metro_m__rel_p7_z_iqr",
        "dist_metro_m__rel_p8_diff_med",
        "dist_metro_m__rel_p8_ratio_med",
        "dist_metro_m__rel_p8_z_iqr",
    }
    assert expected_extra.issubset(set(out.columns))
    # Original columns preserved.
    assert {"object_id", "lat", "lon", "dist_metro_m"}.issubset(set(out.columns))
    assert out.height == 3


def test_three_objects_in_same_parent_yield_correct_relatives() -> None:
    """Three objects sharing one parent at res=7 — median is 200, IQR = 100.
    For F=100: diff=-100, ratio=0.5, z=-1.0; for F=300: diff=+100, ratio=1.5, z=+1.0.
    """
    objects = _df(
        [
            {"object_id": "a", "lat": _KAZAN_CENTER[0], "lon": _KAZAN_CENTER[1], "dist_metro_m": 100.0},
            {"object_id": "b", "lat": _NEARBY[0], "lon": _NEARBY[1], "dist_metro_m": 200.0},
            {"object_id": "c", "lat": _NEARBY2[0], "lon": _NEARBY2[1], "dist_metro_m": 300.0},
        ]
    )

    out = compute_relative_features(
        objects, parent_resolutions=[7], feature_columns=["dist_metro_m"]
    ).sort("object_id")

    # Sanity: all three share the res=7 parent.
    parents = out["parent_h3_p7"].to_list()
    assert parents[0] == parents[1] == parents[2]
    assert out["count_p7"].to_list() == [3, 3, 3]

    diffs = out["dist_metro_m__rel_p7_diff_med"].to_list()
    ratios = out["dist_metro_m__rel_p7_ratio_med"].to_list()
    z_iqrs = out["dist_metro_m__rel_p7_z_iqr"].to_list()

    # median = 200, IQR = p75 - p25 = 250 - 150 = 100 (linear interp on n=3).
    assert diffs == [-100.0, 0.0, 100.0]
    assert ratios == [0.5, 1.0, 1.5]
    assert z_iqrs[0] == -1.0
    assert z_iqrs[1] == 0.0
    assert z_iqrs[2] == 1.0


def test_isolated_object_yields_self_relative_and_count_one() -> None:
    """An object alone in its parent: median = its own value → diff=0, ratio=1.
    IQR = 0 → z_iqr is NaN (we never silently substitute a finite value)."""
    far_lat, far_lon = 55.5000, 49.5500  # different res=7 parent than the cluster above
    objects = _df(
        [
            {"object_id": "lonely", "lat": far_lat, "lon": far_lon, "dist_metro_m": 1234.0},
            {"object_id": "a", "lat": _KAZAN_CENTER[0], "lon": _KAZAN_CENTER[1], "dist_metro_m": 100.0},
            {"object_id": "b", "lat": _NEARBY[0], "lon": _NEARBY[1], "dist_metro_m": 200.0},
        ]
    )

    out = compute_relative_features(
        objects, parent_resolutions=[7], feature_columns=["dist_metro_m"]
    ).sort("object_id")

    lonely = out.filter(pl.col("object_id") == "lonely").to_dicts()[0]
    assert lonely["count_p7"] == 1
    assert lonely["dist_metro_m__rel_p7_diff_med"] == 0.0
    assert lonely["dist_metro_m__rel_p7_ratio_med"] == 1.0
    assert math.isnan(lonely["dist_metro_m__rel_p7_z_iqr"])


def test_zero_median_yields_nan_ratio_not_inf() -> None:
    """When the parent median is 0, the ratio would be inf — but CatBoost
    handles NaN, not inf cleanly (ADR-0011 fix). Use NaN."""
    objects = _df(
        [
            {"object_id": "a", "lat": _KAZAN_CENTER[0], "lon": _KAZAN_CENTER[1], "dist_metro_m": 0.0},
            {"object_id": "b", "lat": _NEARBY[0], "lon": _NEARBY[1], "dist_metro_m": 0.0},
            {"object_id": "c", "lat": _NEARBY2[0], "lon": _NEARBY2[1], "dist_metro_m": 50.0},
        ]
    )

    out = compute_relative_features(
        objects, parent_resolutions=[7], feature_columns=["dist_metro_m"]
    ).sort("object_id")

    ratios = out["dist_metro_m__rel_p7_ratio_med"].to_list()
    # Median of [0,0,50] = 0 → all ratios should be NaN, none inf.
    assert all(math.isnan(r) for r in ratios), ratios
    # And no infinite values leaked through.
    assert not any(math.isinf(r) for r in ratios if r is not None and not math.isnan(r))


def test_multiple_features_each_get_independent_relatives() -> None:
    objects = pl.DataFrame(
        [
            {"object_id": "a", "lat": _KAZAN_CENTER[0], "lon": _KAZAN_CENTER[1],
             "dist_metro_m": 100.0, "year_built": 1980},
            {"object_id": "b", "lat": _NEARBY[0], "lon": _NEARBY[1],
             "dist_metro_m": 200.0, "year_built": 2000},
            {"object_id": "c", "lat": _NEARBY2[0], "lon": _NEARBY2[1],
             "dist_metro_m": 300.0, "year_built": 2020},
        ],
        schema={
            "object_id": pl.Utf8, "lat": pl.Float64, "lon": pl.Float64,
            "dist_metro_m": pl.Float64, "year_built": pl.Int64,
        },
    )

    out = compute_relative_features(
        objects,
        parent_resolutions=[7],
        feature_columns=["dist_metro_m", "year_built"],
    )

    for col in (
        "dist_metro_m__rel_p7_diff_med", "dist_metro_m__rel_p7_ratio_med", "dist_metro_m__rel_p7_z_iqr",
        "year_built__rel_p7_diff_med", "year_built__rel_p7_ratio_med", "year_built__rel_p7_z_iqr",
    ):
        assert col in out.columns


def test_empty_objects_returns_empty_with_correct_schema() -> None:
    objects = _df([])

    out = compute_relative_features(
        objects, parent_resolutions=[7], feature_columns=["dist_metro_m"]
    )

    assert out.is_empty()
    for col in (
        "parent_h3_p7", "count_p7",
        "dist_metro_m__rel_p7_diff_med",
        "dist_metro_m__rel_p7_ratio_med",
        "dist_metro_m__rel_p7_z_iqr",
    ):
        assert col in out.columns


def test_null_in_source_feature_propagates_only_to_that_rows_relatives() -> None:
    """A row with NaN in the source feature gets NaN in its derivatives,
    but must not poison the parent median computed for the other rows.
    """
    objects = pl.DataFrame(
        [
            {"object_id": "a", "lat": _KAZAN_CENTER[0], "lon": _KAZAN_CENTER[1], "dist_metro_m": 100.0},
            {"object_id": "b", "lat": _NEARBY[0], "lon": _NEARBY[1], "dist_metro_m": None},
            {"object_id": "c", "lat": _NEARBY2[0], "lon": _NEARBY2[1], "dist_metro_m": 300.0},
        ],
        schema={
            "object_id": pl.Utf8, "lat": pl.Float64, "lon": pl.Float64, "dist_metro_m": pl.Float64,
        },
    )

    out = compute_relative_features(
        objects, parent_resolutions=[7], feature_columns=["dist_metro_m"]
    ).sort("object_id")

    diffs = out["dist_metro_m__rel_p7_diff_med"].to_list()
    # Median of [100, 300] (ignoring None) = 200. So a → -100, c → +100.
    # The None row keeps its derivatives as None/NaN.
    assert diffs[0] == -100.0
    assert diffs[1] is None or (
        isinstance(diffs[1], float) and math.isnan(diffs[1])
    )
    assert diffs[2] == 100.0


def test_parent_h3_index_matches_h3_library() -> None:
    """parent_h3_p{R} must equal h3.latlng_to_cell(lat, lon, R) — no
    custom rounding, no projection drift."""
    lat, lon = _KAZAN_CENTER
    objects = _df([{"object_id": "a", "lat": lat, "lon": lon, "dist_metro_m": 100.0}])

    out = compute_relative_features(
        objects, parent_resolutions=[7, 8], feature_columns=["dist_metro_m"]
    )

    expected_p7 = h3.latlng_to_cell(lat, lon, 7)
    expected_p8 = h3.latlng_to_cell(lat, lon, 8)
    # h3-py returns int for the cell index; we compare numerically.
    assert int(out["parent_h3_p7"][0]) == expected_p7
    assert int(out["parent_h3_p8"][0]) == expected_p8


def test_count_p_is_per_resolution_not_per_feature() -> None:
    """count_p{R} appears exactly once per resolution, regardless of how
    many feature columns are processed."""
    objects = pl.DataFrame(
        [
            {"object_id": "a", "lat": _KAZAN_CENTER[0], "lon": _KAZAN_CENTER[1],
             "dist_metro_m": 100.0, "year_built": 1980, "levels": 9},
        ],
        schema={
            "object_id": pl.Utf8, "lat": pl.Float64, "lon": pl.Float64,
            "dist_metro_m": pl.Float64, "year_built": pl.Int64, "levels": pl.Int64,
        },
    )

    out = compute_relative_features(
        objects,
        parent_resolutions=[7, 8],
        feature_columns=["dist_metro_m", "year_built", "levels"],
    )

    count_cols = [c for c in out.columns if c.startswith("count_p")]
    assert sorted(count_cols) == ["count_p7", "count_p8"]


def test_all_relatives_are_finite_or_nan_never_inf() -> None:
    """Across a small synthetic set, no derivative should be ±inf —
    consistent with ADR-0011 inf-clamp policy. NaN is fine."""
    objects = _df(
        [
            {"object_id": "a", "lat": _KAZAN_CENTER[0], "lon": _KAZAN_CENTER[1], "dist_metro_m": 0.0},
            {"object_id": "b", "lat": _NEARBY[0], "lon": _NEARBY[1], "dist_metro_m": 50.0},
            {"object_id": "c", "lat": _NEARBY2[0], "lon": _NEARBY2[1], "dist_metro_m": 0.0},
        ]
    )

    out = compute_relative_features(
        objects, parent_resolutions=[7], feature_columns=["dist_metro_m"]
    )

    for col in (
        "dist_metro_m__rel_p7_diff_med",
        "dist_metro_m__rel_p7_ratio_med",
        "dist_metro_m__rel_p7_z_iqr",
    ):
        arr = np.array(out[col].to_list(), dtype=np.float64)
        assert not np.isinf(arr).any(), (col, arr)
