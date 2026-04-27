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
    _write_oof(older, [{"object_id": "old", "lat": 0.0, "lon": 0.0, "fold_id": 0, "y_true": 1.0, "y_pred_oof": 1.1}])
    _write_oof(newer, [{"object_id": "new", "lat": 0.0, "lon": 0.0, "fold_id": 0, "y_true": 2.0, "y_pred_oof": 2.2}])

    df = LocalOofPredictionsReader(tmp_path).load_latest(AssetClass.APARTMENT)

    assert df.height == 1
    assert df["object_id"][0] == "new"


def test_load_latest_filters_by_class_prefix(tmp_path: Path) -> None:
    apartment_run = tmp_path / "catboost-object-apartment_20260101T120000Z"
    house_run = tmp_path / "catboost-object-house_20260201T120000Z"
    _write_oof(
        apartment_run, [{"object_id": "a", "lat": 0.0, "lon": 0.0, "fold_id": 0, "y_true": 1.0, "y_pred_oof": 1.0}]
    )
    _write_oof(house_run, [{"object_id": "h", "lat": 0.0, "lon": 0.0, "fold_id": 0, "y_true": 2.0, "y_pred_oof": 2.0}])

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
    df = LocalOofPredictionsReader(tmp_path / "does-not-exist").load_latest(AssetClass.APARTMENT)
    assert df.is_empty()


# --- ADR-0016 quartet support: per-model OOF reading ----------------


def _write_quartet_run(run_dir: Path, model_to_rows: dict[str, list[dict[str, object]]]) -> None:
    """Write per-model oof parquets the way TrainQuartet emits them:
    one file per model named ``{model}_oof_predictions.parquet``."""
    run_dir.mkdir(parents=True, exist_ok=True)
    for model, rows in model_to_rows.items():
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
        ).write_parquet(run_dir / f"{model}_oof_predictions.parquet")


def test_load_latest_catboost_picks_quartet_run_when_newer(tmp_path: Path) -> None:
    """A quartet run is newer than the catboost-only run; the catboost
    OOF should come from the quartet's catboost_oof_predictions.parquet."""
    _write_oof(
        tmp_path / "catboost-object-apartment_20260101T120000000000Z",
        [{"object_id": "old-cb", "lat": 0.0, "lon": 0.0, "fold_id": 0, "y_true": 1.0, "y_pred_oof": 1.0}],
    )
    _write_quartet_run(
        tmp_path / "quartet-object-apartment_20260601T120000000000Z",
        {
            "catboost": [{"object_id": "q-cb", "lat": 0.0, "lon": 0.0, "fold_id": 0, "y_true": 5.0, "y_pred_oof": 5.5}],
            "ebm": [{"object_id": "q-ebm", "lat": 0.0, "lon": 0.0, "fold_id": 0, "y_true": 5.0, "y_pred_oof": 5.7}],
        },
    )

    df = LocalOofPredictionsReader(tmp_path).load_latest(AssetClass.APARTMENT, model="catboost")
    assert df.height == 1
    assert df["object_id"][0] == "q-cb"


def test_load_latest_catboost_falls_back_to_legacy_run(tmp_path: Path) -> None:
    """When only the catboost-only run exists (no quartet yet), the
    legacy oof_predictions.parquet path must still work."""
    _write_oof(
        tmp_path / "catboost-object-apartment_20260101T120000000000Z",
        [{"object_id": "legacy", "lat": 0.0, "lon": 0.0, "fold_id": 0, "y_true": 1.0, "y_pred_oof": 1.1}],
    )
    df = LocalOofPredictionsReader(tmp_path).load_latest(AssetClass.APARTMENT, model="catboost")
    assert df.height == 1
    assert df["object_id"][0] == "legacy"


def test_load_latest_ebm_reads_quartet_run(tmp_path: Path) -> None:
    _write_quartet_run(
        tmp_path / "quartet-object-apartment_20260601T120000000000Z",
        {
            "catboost": [],
            "ebm": [{"object_id": "ebm-1", "lat": 0.0, "lon": 0.0, "fold_id": 0, "y_true": 5.0, "y_pred_oof": 5.7}],
            "grey_tree": [{"object_id": "g-1", "lat": 0.0, "lon": 0.0, "fold_id": 0, "y_true": 5.0, "y_pred_oof": 5.6}],
            "naive_linear": [
                {"object_id": "n-1", "lat": 0.0, "lon": 0.0, "fold_id": 0, "y_true": 5.0, "y_pred_oof": 5.0}
            ],
        },
    )
    reader = LocalOofPredictionsReader(tmp_path)
    assert reader.load_latest(AssetClass.APARTMENT, model="ebm")["object_id"][0] == "ebm-1"
    assert reader.load_latest(AssetClass.APARTMENT, model="grey_tree")["object_id"][0] == "g-1"
    assert reader.load_latest(AssetClass.APARTMENT, model="naive_linear")["object_id"][0] == "n-1"


def test_load_latest_ebm_returns_empty_when_only_legacy_catboost_run(
    tmp_path: Path,
) -> None:
    """No quartet run yet → ebm/grey/naive should yield empty (typed)
    DataFrame; the UI treats this as 'predictions unavailable'."""
    _write_oof(
        tmp_path / "catboost-object-apartment_20260101T120000000000Z",
        [{"object_id": "x", "lat": 0.0, "lon": 0.0, "fold_id": 0, "y_true": 1.0, "y_pred_oof": 1.0}],
    )
    df = LocalOofPredictionsReader(tmp_path).load_latest(AssetClass.APARTMENT, model="ebm")
    assert df.is_empty()
    assert set(df.columns) >= {"object_id", "y_true", "y_pred_oof"}
