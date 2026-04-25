"""Tests for LocalOofPredictionsReader.

Picks the lexicographically largest run directory matching
``catboost-object-{class}_*`` and reads ``oof_predictions.parquet``
from inside. Returns an empty DataFrame on missing path / missing run /
missing artifact — never raises.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from kadastra.adapters.local_oof_predictions_reader import (
    LocalOofPredictionsReader,
)
from kadastra.domain.asset_class import AssetClass


def _write_oof(run_dir: Path, rows: list[dict[str, object]]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        rows,
        schema={
            "object_id": pl.Utf8,
            "lat": pl.Float64,
            "lon": pl.Float64,
            "fold_id": pl.Int64,
            "y_true": pl.Float64,
            "y_pred_oof": pl.Float64,
        },
    ).write_parquet(run_dir / "oof_predictions.parquet")


def test_load_latest_returns_data_from_newest_run(tmp_path: Path) -> None:
    older = tmp_path / "catboost-object-apartment_20260101T120000000000Z"
    newer = tmp_path / "catboost-object-apartment_20260201T120000000000Z"
    _write_oof(older, [
        {"object_id": "old", "lat": 0.0, "lon": 0.0, "fold_id": 0, "y_true": 1.0, "y_pred_oof": 1.1}
    ])
    _write_oof(newer, [
        {"object_id": "new", "lat": 0.0, "lon": 0.0, "fold_id": 0, "y_true": 2.0, "y_pred_oof": 2.2}
    ])

    df = LocalOofPredictionsReader(tmp_path).load_latest(AssetClass.APARTMENT)

    assert df.height == 1
    assert df["object_id"][0] == "new"


def test_load_latest_filters_by_class_prefix(tmp_path: Path) -> None:
    apartment_run = tmp_path / "catboost-object-apartment_20260101T120000Z"
    house_run = tmp_path / "catboost-object-house_20260201T120000Z"
    _write_oof(apartment_run, [
        {"object_id": "a", "lat": 0.0, "lon": 0.0, "fold_id": 0, "y_true": 1.0, "y_pred_oof": 1.0}
    ])
    _write_oof(house_run, [
        {"object_id": "h", "lat": 0.0, "lon": 0.0, "fold_id": 0, "y_true": 2.0, "y_pred_oof": 2.0}
    ])

    df = LocalOofPredictionsReader(tmp_path).load_latest(AssetClass.APARTMENT)

    assert df.height == 1
    assert df["object_id"][0] == "a"


def test_load_latest_returns_empty_when_no_runs_match(tmp_path: Path) -> None:
    # Only a different-class run is present.
    _write_oof(
        tmp_path / "catboost-object-house_20260201T120000Z",
        [{"object_id": "h", "lat": 0.0, "lon": 0.0, "fold_id": 0, "y_true": 1.0, "y_pred_oof": 1.0}],
    )
    df = LocalOofPredictionsReader(tmp_path).load_latest(AssetClass.APARTMENT)
    assert df.is_empty()
    assert set(df.columns) == {"object_id", "lat", "lon", "fold_id", "y_true", "y_pred_oof"}


def test_load_latest_returns_empty_when_artifact_missing(tmp_path: Path) -> None:
    # Run dir exists but no oof_predictions.parquet inside — registered
    # before OOF artifacts were introduced.
    (tmp_path / "catboost-object-apartment_20260101T120000Z").mkdir()
    df = LocalOofPredictionsReader(tmp_path).load_latest(AssetClass.APARTMENT)
    assert df.is_empty()


def test_load_latest_returns_empty_when_base_path_missing(tmp_path: Path) -> None:
    df = LocalOofPredictionsReader(tmp_path / "does-not-exist").load_latest(
        AssetClass.APARTMENT
    )
    assert df.is_empty()
