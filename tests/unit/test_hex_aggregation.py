"""Tests for aggregate_objects_to_hex.

Per-hex aggregator that turns per-object gold + OOF predictions into
a (resolution, h3_index, asset_class) wide table for the map UI.
"""

from __future__ import annotations

import h3
import polars as pl

from kadastra.etl.hex_aggregation import aggregate_objects_to_hex

KAZAN_LAT, KAZAN_LON = 55.7887, 49.1221


def _objects(rows: list[dict[str, object]]) -> pl.DataFrame:
    schema = {
        "object_id": pl.Utf8,
        "asset_class": pl.Utf8,
        "lat": pl.Float64,
        "lon": pl.Float64,
        "synthetic_target_rub_per_m2": pl.Float64,
        "y_pred_oof": pl.Float64,
        "levels": pl.Int64,
        "intra_city_raion": pl.Utf8,
    }
    return pl.DataFrame(rows, schema=schema)


def test_groups_by_hex_and_asset_class() -> None:
    """Two apartments in the same hex should produce one apartment row;
    add a house in the same hex → second row. Plus an "all" roll-up."""
    rows = [
        {
            "object_id": "a1", "asset_class": "apartment",
            "lat": KAZAN_LAT, "lon": KAZAN_LON,
            "synthetic_target_rub_per_m2": 100_000.0,
            "y_pred_oof": 95_000.0, "levels": 5,
            "intra_city_raion": "Советский",
        },
        {
            "object_id": "a2", "asset_class": "apartment",
            "lat": KAZAN_LAT + 1e-5, "lon": KAZAN_LON + 1e-5,
            "synthetic_target_rub_per_m2": 110_000.0,
            "y_pred_oof": 100_000.0, "levels": 9,
            "intra_city_raion": "Советский",
        },
        {
            "object_id": "h1", "asset_class": "house",
            "lat": KAZAN_LAT + 2e-5, "lon": KAZAN_LON + 2e-5,
            "synthetic_target_rub_per_m2": 50_000.0,
            "y_pred_oof": 60_000.0, "levels": 2,
            "intra_city_raion": "Советский",
        },
    ]
    out = aggregate_objects_to_hex(_objects(rows), resolution=10)

    classes = sorted(out["asset_class"].unique().to_list())
    assert "apartment" in classes
    assert "house" in classes
    assert "all" in classes

    # Each (h3, class) tuple unique → 3 rows for this single-hex case
    # (apartment, house, all).
    assert out.height == 3


def test_median_target_and_pred_per_hex() -> None:
    rows = [
        {
            "object_id": f"a{i}", "asset_class": "apartment",
            "lat": KAZAN_LAT + 1e-5 * i, "lon": KAZAN_LON + 1e-5 * i,
            "synthetic_target_rub_per_m2": float(t),
            "y_pred_oof": float(p), "levels": 5,
            "intra_city_raion": "Советский",
        }
        for i, (t, p) in enumerate([(100, 90), (200, 180), (300, 280)])
    ]
    out = aggregate_objects_to_hex(_objects(rows), resolution=10)
    apt = out.filter(pl.col("asset_class") == "apartment").row(0, named=True)
    assert apt["count"] == 3
    assert apt["median_target_rub_per_m2"] == 200.0
    assert apt["median_pred_oof_rub_per_m2"] == 180.0
    # Residual = pred − true → median of (-10, -20, -20) = -20
    assert apt["median_residual_rub_per_m2"] == -20.0


def test_dominant_intra_raion() -> None:
    rows = [
        {
            "object_id": f"a{i}", "asset_class": "apartment",
            "lat": KAZAN_LAT + 1e-5 * i, "lon": KAZAN_LON + 1e-5 * i,
            "synthetic_target_rub_per_m2": 100.0,
            "y_pred_oof": 100.0, "levels": 5,
            "intra_city_raion": raion,
        }
        for i, raion in enumerate(
            ["Советский", "Советский", "Вахитовский"]
        )
    ]
    out = aggregate_objects_to_hex(_objects(rows), resolution=10)
    apt = out.filter(pl.col("asset_class") == "apartment").row(0, named=True)
    assert apt["dominant_intra_city_raion"] == "Советский"


def test_works_without_oof_pred_column() -> None:
    """When OOF artifacts are missing, ``y_pred_oof`` is absent from
    the input. The aggregator must still produce all expected columns
    with null prediction medians."""
    schema = {
        "object_id": pl.Utf8,
        "asset_class": pl.Utf8,
        "lat": pl.Float64,
        "lon": pl.Float64,
        "synthetic_target_rub_per_m2": pl.Float64,
    }
    df = pl.DataFrame(
        [
            {
                "object_id": "a1", "asset_class": "apartment",
                "lat": KAZAN_LAT, "lon": KAZAN_LON,
                "synthetic_target_rub_per_m2": 100.0,
            }
        ],
        schema=schema,
    )
    out = aggregate_objects_to_hex(df, resolution=10)
    apt = out.filter(pl.col("asset_class") == "apartment").row(0, named=True)
    assert apt["median_target_rub_per_m2"] == 100.0
    assert apt["median_pred_oof_rub_per_m2"] is None


def test_resolution_column_set_correctly() -> None:
    rows = [
        {
            "object_id": "a1", "asset_class": "apartment",
            "lat": KAZAN_LAT, "lon": KAZAN_LON,
            "synthetic_target_rub_per_m2": 100.0, "y_pred_oof": 100.0,
            "levels": 5, "intra_city_raion": "Советский",
        }
    ]
    out = aggregate_objects_to_hex(_objects(rows), resolution=8)
    assert (out["resolution"] == 8).all()
    # Sanity: the produced h3_index is a valid res-8 cell.
    assert h3.get_resolution(out["h3_index"][0]) == 8


def test_empty_input_returns_empty_with_schema() -> None:
    df = pl.DataFrame(
        schema={
            "object_id": pl.Utf8,
            "asset_class": pl.Utf8,
            "lat": pl.Float64,
            "lon": pl.Float64,
            "synthetic_target_rub_per_m2": pl.Float64,
        }
    )
    out = aggregate_objects_to_hex(df, resolution=10)
    assert out.is_empty()
    assert "h3_index" in out.columns
    assert "asset_class" in out.columns
    assert "median_target_rub_per_m2" in out.columns
