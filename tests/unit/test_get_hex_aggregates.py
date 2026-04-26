"""Tests for GetHexAggregates.

Reads per-hex aggregates parquet and returns ``[{"hex", "value"}]``
filtered by asset_class + feature. Numeric and categorical features
are both supported (the map UI decides the rendering style).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from kadastra.usecases.get_hex_aggregates import GetHexAggregates


def _write_aggregates(
    tmp_path: Path,
    region: str,
    resolution: int,
    *,
    model: str = "catboost",
    pred_8a: float | None = 95_000.0,
) -> None:
    path = (
        tmp_path
        / f"region={region}"
        / f"resolution={resolution}"
        / f"model={model}"
        / "data.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(
        [
            {"h3_index": "8a", "resolution": resolution, "asset_class": "apartment",
             "count": 10, "median_target_rub_per_m2": 100_000.0,
             "median_pred_oof_rub_per_m2": pred_8a,
             "dominant_intra_city_raion": "Советский"},
            {"h3_index": "8b", "resolution": resolution, "asset_class": "apartment",
             "count": 5, "median_target_rub_per_m2": 80_000.0,
             "median_pred_oof_rub_per_m2": None,
             "dominant_intra_city_raion": "Вахитовский"},
            {"h3_index": "8a", "resolution": resolution, "asset_class": "all",
             "count": 30, "median_target_rub_per_m2": 50_000.0,
             "median_pred_oof_rub_per_m2": 50_000.0,
             "dominant_intra_city_raion": "Советский"},
        ],
        schema={
            "h3_index": pl.Utf8, "resolution": pl.Int32, "asset_class": pl.Utf8,
            "count": pl.UInt32, "median_target_rub_per_m2": pl.Float64,
            "median_pred_oof_rub_per_m2": pl.Float64,
            "dominant_intra_city_raion": pl.Utf8,
        },
    )
    df.write_parquet(path)


def test_filters_by_asset_class(tmp_path: Path) -> None:
    _write_aggregates(tmp_path, "RU-KAZAN-AGG", 8)
    out = GetHexAggregates(tmp_path).execute(
        "RU-KAZAN-AGG", 8, asset_class="apartment", feature="count"
    )
    hexes = sorted(r["hex"] for r in out)
    assert hexes == ["8a", "8b"]


def test_returns_numeric_value_for_numeric_feature(tmp_path: Path) -> None:
    _write_aggregates(tmp_path, "RU-KAZAN-AGG", 8)
    out = GetHexAggregates(tmp_path).execute(
        "RU-KAZAN-AGG", 8, asset_class="apartment", feature="median_target_rub_per_m2"
    )
    by_hex = {r["hex"]: r["value"] for r in out}
    assert by_hex["8a"] == 100_000.0
    assert by_hex["8b"] == 80_000.0


def test_drops_null_values(tmp_path: Path) -> None:
    _write_aggregates(tmp_path, "RU-KAZAN-AGG", 8)
    out = GetHexAggregates(tmp_path).execute(
        "RU-KAZAN-AGG", 8, asset_class="apartment",
        feature="median_pred_oof_rub_per_m2",
    )
    # 8b has null median_pred → must be filtered out
    hexes = [r["hex"] for r in out]
    assert hexes == ["8a"]


def test_categorical_feature_returns_string_value(tmp_path: Path) -> None:
    _write_aggregates(tmp_path, "RU-KAZAN-AGG", 8)
    out = GetHexAggregates(tmp_path).execute(
        "RU-KAZAN-AGG", 8, asset_class="apartment",
        feature="dominant_intra_city_raion",
    )
    by_hex = {r["hex"]: r["value"] for r in out}
    assert by_hex["8a"] == "Советский"
    assert by_hex["8b"] == "Вахитовский"


def test_unknown_feature_raises_keyerror(tmp_path: Path) -> None:
    _write_aggregates(tmp_path, "RU-KAZAN-AGG", 8)
    with pytest.raises(KeyError, match="not in hex_aggregates"):
        GetHexAggregates(tmp_path).execute(
            "RU-KAZAN-AGG", 8, asset_class="apartment", feature="bogus"
        )


def test_missing_parquet_raises_filenotfound(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        GetHexAggregates(tmp_path).execute(
            "RU-KAZAN-AGG", 8, asset_class="apartment", feature="count"
        )


def test_model_param_routes_to_correct_partition(tmp_path: Path) -> None:
    """ADR-0016: ``GetHexAggregates`` reads ``model={model}`` partition.
    Two partitions with different ``median_pred_oof`` must yield two
    different sets of values for the same hex / asset_class / feature."""
    _write_aggregates(tmp_path, "RU-KAZAN-AGG", 8, model="catboost", pred_8a=95_000.0)
    _write_aggregates(tmp_path, "RU-KAZAN-AGG", 8, model="ebm", pred_8a=80_000.0)

    cb_out = GetHexAggregates(tmp_path).execute(
        "RU-KAZAN-AGG", 8, asset_class="apartment",
        feature="median_pred_oof_rub_per_m2", model="catboost",
    )
    ebm_out = GetHexAggregates(tmp_path).execute(
        "RU-KAZAN-AGG", 8, asset_class="apartment",
        feature="median_pred_oof_rub_per_m2", model="ebm",
    )
    cb_by_hex = {r["hex"]: r["value"] for r in cb_out}
    ebm_by_hex = {r["hex"]: r["value"] for r in ebm_out}
    assert cb_by_hex["8a"] == 95_000.0
    assert ebm_by_hex["8a"] == 80_000.0


def test_default_model_is_catboost(tmp_path: Path) -> None:
    """No ``model=`` kwarg → must read ``model=catboost`` partition.
    Keeps the existing UI working before it's wired to send
    ``?model=…`` to the API."""
    _write_aggregates(tmp_path, "RU-KAZAN-AGG", 8, model="catboost", pred_8a=42_000.0)
    out = GetHexAggregates(tmp_path).execute(
        "RU-KAZAN-AGG", 8, asset_class="apartment",
        feature="median_pred_oof_rub_per_m2",
    )
    assert {r["hex"]: r["value"] for r in out}["8a"] == 42_000.0


def test_missing_model_partition_raises_filenotfound(tmp_path: Path) -> None:
    """If only catboost partition exists, ?model=ebm must surface as
    404 (FileNotFoundError → API maps to 404), not silently fall back
    to catboost."""
    _write_aggregates(tmp_path, "RU-KAZAN-AGG", 8, model="catboost")
    with pytest.raises(FileNotFoundError):
        GetHexAggregates(tmp_path).execute(
            "RU-KAZAN-AGG", 8, asset_class="apartment",
            feature="count", model="ebm",
        )
