"""Smoke test for TrainQuartet (ADR-0016).

Synthetic per-class gold (small, 200 rows around Kazan), runs the full
quartet pipeline (Black/White/Grey/Naive), asserts that:

- the run is logged to ``ModelRegistryPort`` once,
- ``quartet_metrics.json`` artifact carries entries for all four
  models + a ``loss_on_simplification`` block,
- per-model ``*_oof_predictions.parquet`` artifacts are written and
  shape-correct,
- Grey Box fidelity (R² to Black) is reported.

We don't pin specific MAPE values — random synthetic data + small
sample = high variance. The test is structural: «did all four models
fit, predict, and end up in the artifact?»
"""

from __future__ import annotations

import io
import json
import pickle
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
from catboost import CatBoostRegressor

from kadastra.domain.asset_class import AssetClass
from kadastra.ml.train import CatBoostParams
from kadastra.usecases.train_quartet import TrainQuartet


class _FakeReader:
    def __init__(self, df: pl.DataFrame) -> None:
        self._df = df

    def load(self, region_code: str, asset_class: AssetClass) -> pl.DataFrame:
        return self._df


class _FakeRegistry:
    def __init__(self) -> None:
        self.runs: list[dict[str, Any]] = []

    def log_run(
        self,
        *,
        run_name: str,
        params: Mapping[str, Any],
        metrics: Mapping[str, float],
        model: CatBoostRegressor,
        artifacts: Mapping[str, bytes] | None = None,
    ) -> str:
        self.runs.append(
            {
                "run_name": run_name,
                "params": dict(params),
                "metrics": dict(metrics),
                "model": model,
                "artifacts": dict(artifacts or {}),
            }
        )
        return f"{run_name}_run0"


def _synth_gold(n: int = 240) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    lat = 55.6 + rng.uniform(0.0, 0.5, n)
    lon = 49.0 + rng.uniform(0.0, 0.5, n)
    area = rng.uniform(40.0, 200.0, n)
    levels = rng.integers(1, 25, n)
    year = rng.integers(1960, 2024, n)
    raion = rng.choice(["Советский", "Вахитовский", "Приволжский"], n)
    okrug = np.full(n, "город Казань", dtype=object)
    # Target structured by (raion, area, levels) so EBM/Naive can learn
    # something instead of fitting noise.
    raion_offset = np.array([{"Советский": 100_000, "Вахитовский": 130_000, "Приволжский": 80_000}[r] for r in raion])
    target = raion_offset + area * 50.0 - levels * 200.0 + rng.normal(0.0, 5_000, n)
    return pl.DataFrame(
        {
            "object_id": [f"way/{i}" for i in range(n)],
            "asset_class": ["apartment"] * n,
            "lat": lat.astype(np.float64),
            "lon": lon.astype(np.float64),
            "area_m2": area.astype(np.float64),
            "levels": levels.astype(np.int64),
            "year_built": year.astype(np.int64),
            "intra_city_raion": raion.astype(str),
            "mun_okrug_name": okrug.astype(str),
            "synthetic_target_rub_per_m2": target.astype(np.float64),
            "cost_value_rub": (target * area).astype(np.float64),
        }
    )


def test_quartet_runs_and_logs_full_metrics() -> None:
    reader = _FakeReader(_synth_gold())
    registry = _FakeRegistry()
    usecase = TrainQuartet(
        reader=reader,
        model_registry=registry,
        catboost_params=CatBoostParams(iterations=80, learning_rate=0.1, depth=4, seed=42),
        ebm_max_bins=64,
        ebm_interactions=0,
        grey_tree_max_depth=6,
        n_splits=3,
        parent_resolution=6,
    )
    run_id = usecase.execute("RU-KAZAN-AGG", AssetClass.APARTMENT)

    assert run_id.startswith("quartet-object-apartment")
    assert len(registry.runs) == 1
    run = registry.runs[0]
    artifacts = run["artifacts"]

    # quartet_metrics.json contains all four models + simplification loss.
    raw = artifacts["quartet_metrics.json"]
    payload = json.loads(raw.decode("utf-8"))
    assert payload["asset_class"] == "apartment"
    for model in ("catboost", "ebm", "grey_tree", "naive_linear"):
        assert model in payload["models"]
        m = payload["models"][model]
        for key in ("mean_mae", "mean_rmse", "mean_mape", "mean_spearman", "wape"):
            assert key in m
            assert isinstance(m[key], (int, float))
    # Grey carries a fidelity-to-black field.
    assert "fidelity_r2_to_catboost" in payload["models"]["grey_tree"]
    # Loss-on-simplification block in pp.
    los = payload["loss_on_simplification"]
    assert "ebm_minus_catboost_mape_pp" in los
    assert "naive_minus_catboost_mape_pp" in los

    # Per-model OOF parquets exist and shape-correctly.
    for model in ("catboost", "ebm", "grey_tree", "naive_linear"):
        name = f"{model}_oof_predictions.parquet"
        assert name in artifacts
        df = pl.read_parquet(io.BytesIO(artifacts[name]))
        assert set(df.columns) >= {"object_id", "lat", "lon", "fold_id", "y_true", "y_pred_oof"}
        assert df.height == 240


def test_default_keeps_all_simplifier_final_fit_pkl_artifacts() -> None:
    """Baseline: by default the three simplifier final-fit pickles
    (EBM/Grey/Naive) end up in artifacts so loaders can deserialize
    full-data models from a registry run."""
    reader = _FakeReader(_synth_gold())
    registry = _FakeRegistry()
    usecase = TrainQuartet(
        reader=reader,
        model_registry=registry,
        catboost_params=CatBoostParams(iterations=40, learning_rate=0.1, depth=4, seed=42),
        ebm_max_bins=32,
        ebm_interactions=0,
        grey_tree_max_depth=6,
        n_splits=3,
        parent_resolution=6,
    )
    usecase.execute("RU-KAZAN-AGG", AssetClass.APARTMENT)
    artifacts = registry.runs[0]["artifacts"]
    assert "ebm_model.pkl" in artifacts
    assert "grey_tree_model.pkl" in artifacts
    assert "naive_linear_model.pkl" in artifacts


def test_skip_final_simplifier_fits_omits_pkl_artifacts() -> None:
    """S2: when skip_final_simplifier_fits=True, the Grey/Naive full-data
    refits are skipped (no consumer reads their *_model.pkl). The EBM
    (White Box) final fit is ALWAYS kept — the inspector's explanation
    endpoint loads ebm_model.pkl. CatBoost final fit stays because the
    registry contract still requires a primary model.

    OOF parquets and quartet_metrics.json must remain identical in
    structure — only the Grey/Naive *_model.pkl artifacts are dropped.
    """
    reader = _FakeReader(_synth_gold())
    registry = _FakeRegistry()
    usecase = TrainQuartet(
        reader=reader,
        model_registry=registry,
        catboost_params=CatBoostParams(iterations=40, learning_rate=0.1, depth=4, seed=42),
        ebm_max_bins=32,
        ebm_interactions=0,
        grey_tree_max_depth=6,
        n_splits=3,
        parent_resolution=6,
        skip_final_simplifier_fits=True,
    )
    usecase.execute("RU-KAZAN-AGG", AssetClass.APARTMENT)
    artifacts = registry.runs[0]["artifacts"]
    assert "ebm_model.pkl" in artifacts
    assert "grey_tree_model.pkl" not in artifacts
    assert "naive_linear_model.pkl" not in artifacts
    # OOF parquets and metrics still produced.
    for model in ("catboost", "ebm", "grey_tree", "naive_linear"):
        assert f"{model}_oof_predictions.parquet" in artifacts
    assert "quartet_metrics.json" in artifacts


def test_parallel_folds_smoke_runs_and_produces_oof_for_all_models() -> None:
    """S1: when parallel_folds=True, the per-fold loops are dispatched
    via joblib so independent folds can train concurrently. Result
    artifacts must remain shape-identical to sequential mode (same OOF
    columns, same row count, all four models present)."""
    reader = _FakeReader(_synth_gold())
    registry = _FakeRegistry()
    usecase = TrainQuartet(
        reader=reader,
        model_registry=registry,
        catboost_params=CatBoostParams(iterations=40, learning_rate=0.1, depth=4, seed=42),
        ebm_max_bins=32,
        ebm_interactions=0,
        grey_tree_max_depth=6,
        n_splits=3,
        parent_resolution=6,
        parallel_folds=True,
    )
    run_id = usecase.execute("RU-KAZAN-AGG", AssetClass.APARTMENT)
    assert run_id.startswith("quartet-object-apartment")
    artifacts = registry.runs[0]["artifacts"]
    for model in ("catboost", "ebm", "grey_tree", "naive_linear"):
        name = f"{model}_oof_predictions.parquet"
        assert name in artifacts
        df = pl.read_parquet(io.BytesIO(artifacts[name]))
        assert set(df.columns) >= {"object_id", "lat", "lon", "fold_id", "y_true", "y_pred_oof"}
        assert df.height == 240


def test_landplot_trains_on_identity_and_reports_wape() -> None:
    """ADR-0026: WAPE is the cross-class relative metric for landplot —
    stable on the tiny ₽/м² denominator that breaks MAPE. The class trains
    on the raw target (no log transform; log-target was reverted after an
    A/B showed no ranking/aggregate gain)."""
    reader = _FakeReader(_synth_gold())
    registry = _FakeRegistry()
    usecase = TrainQuartet(
        reader=reader,
        model_registry=registry,
        catboost_params=CatBoostParams(iterations=40, learning_rate=0.1, depth=4, seed=42),
        ebm_max_bins=32,
        ebm_interactions=0,
        grey_tree_max_depth=6,
        n_splits=3,
        parent_resolution=6,
    )
    usecase.execute("RU-KAZAN-AGG", AssetClass.LANDPLOT)
    artifacts = registry.runs[0]["artifacts"]
    payload = json.loads(artifacts["quartet_metrics.json"].decode("utf-8"))
    # No target transform — metrics are in the original ₽/м² scale.
    assert "target_transform" not in payload
    for model in ("catboost", "ebm", "grey_tree", "naive_linear"):
        assert "wape" in payload["models"][model]
        assert "rmse_log" not in payload["models"][model]


class _CrashingRegistry(_FakeRegistry):
    """log_run raises — simulates a kill right at the end of execute(),
    leaving the checkpoint dir behind for a resumed run."""

    def log_run(self, **kwargs: Any) -> str:
        raise RuntimeError("simulated crash")


def _make_usecase(
    df: pl.DataFrame,
    registry: _FakeRegistry,
    checkpoint_dir: Path,
    *,
    resume: bool = True,
) -> TrainQuartet:
    return TrainQuartet(
        reader=_FakeReader(df),
        model_registry=registry,
        catboost_params=CatBoostParams(iterations=40, learning_rate=0.1, depth=4, seed=42),
        ebm_max_bins=32,
        ebm_interactions=0,
        grey_tree_max_depth=6,
        n_splits=3,
        parent_resolution=6,
        checkpoint_dir=checkpoint_dir,
        resume=resume,
    )


def test_checkpoints_resume_after_crash(tmp_path: Path) -> None:
    """Crash recovery: a run killed before log_run leaves per-stage
    checkpoints; a restarted run with the same data+params fingerprint
    resumes finished stages (checkpoint files are NOT rewritten) and the
    next successful run produces full artifacts and discards the dir."""
    df = _synth_gold()
    ckpt = tmp_path / "ckpt"

    with pytest.raises(RuntimeError, match="simulated crash"):
        _make_usecase(df, _CrashingRegistry(), ckpt).execute("RU-KAZAN-AGG", AssetClass.APARTMENT)

    stage_dir = ckpt / "quartet-object-apartment"
    assert len(list(stage_dir.glob("pass1_fold_*.pkl"))) == 3
    assert len(list(stage_dir.glob("pass2_fold_*.pkl"))) == 3
    assert (stage_dir / "final_catboost.pkl").exists()
    assert (stage_dir / "final_ebm.pkl").exists()
    mtimes_before = {p.name: p.stat().st_mtime_ns for p in stage_dir.glob("*.pkl")}

    # Second crash: every stage resumes — no checkpoint file is rewritten.
    with pytest.raises(RuntimeError, match="simulated crash"):
        _make_usecase(df, _CrashingRegistry(), ckpt).execute("RU-KAZAN-AGG", AssetClass.APARTMENT)
    for name, mtime in mtimes_before.items():
        assert (stage_dir / name).stat().st_mtime_ns == mtime, f"stage {name} was recomputed"

    # Successful run: full artifacts, checkpoints discarded.
    registry = _FakeRegistry()
    _make_usecase(df, registry, ckpt).execute("RU-KAZAN-AGG", AssetClass.APARTMENT)
    artifacts = registry.runs[0]["artifacts"]
    assert "quartet_metrics.json" in artifacts
    assert "ebm_model.pkl" in artifacts
    assert not stage_dir.exists()


def test_fingerprint_mismatch_ignores_stale_checkpoints(tmp_path: Path) -> None:
    """A checkpoint from a DIFFERENT dataset (target changed) must never
    be resumed: the fingerprint mismatch invalidates it and stages are
    recomputed — the checkpoint payload changes accordingly."""
    df_a = _synth_gold()
    ckpt = tmp_path / "ckpt"
    with pytest.raises(RuntimeError, match="simulated crash"):
        _make_usecase(df_a, _CrashingRegistry(), ckpt).execute("RU-KAZAN-AGG", AssetClass.APARTMENT)
    stage_dir = ckpt / "quartet-object-apartment"
    with (stage_dir / "pass1_fold_0.pkl").open("rb") as fh:
        stale_payload = pickle.load(fh)

    # Same schema, shifted target → different fingerprint.
    df_b = df_a.with_columns((pl.col("synthetic_target_rub_per_m2") * 1.5).alias("synthetic_target_rub_per_m2"))
    with pytest.raises(RuntimeError, match="simulated crash"):
        _make_usecase(df_b, _CrashingRegistry(), ckpt).execute("RU-KAZAN-AGG", AssetClass.APARTMENT)

    with (stage_dir / "pass1_fold_0.pkl").open("rb") as fh:
        fresh_payload = pickle.load(fh)
    assert not np.allclose(stale_payload["cb_pred"], fresh_payload["cb_pred"])


def test_resume_false_recomputes_despite_checkpoints(tmp_path: Path) -> None:
    """resume=False recomputes every stage even when valid checkpoints
    exist (the script's --fresh flag deletes the dir up front instead)."""
    df = _synth_gold()
    ckpt = tmp_path / "ckpt"
    with pytest.raises(RuntimeError, match="simulated crash"):
        _make_usecase(df, _CrashingRegistry(), ckpt).execute("RU-KAZAN-AGG", AssetClass.APARTMENT)
    stage_dir = ckpt / "quartet-object-apartment"
    mtime_before = (stage_dir / "pass1_fold_0.pkl").stat().st_mtime_ns

    time.sleep(0.05)  # keep mtimes distinguishable on coarse filesystems
    with pytest.raises(RuntimeError, match="simulated crash"):
        _make_usecase(df, _CrashingRegistry(), ckpt, resume=False).execute("RU-KAZAN-AGG", AssetClass.APARTMENT)
    assert (stage_dir / "pass1_fold_0.pkl").stat().st_mtime_ns != mtime_before
