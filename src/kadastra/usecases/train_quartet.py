"""TrainQuartet use case (ADR-0016).

Wires together the four QuartetModel adapters (Black/White/Grey/Naive)
and computes per-fold + aggregate metrics on a single set of spatial-CV
folds, so per-fold comparison is honest. Logs everything as a single
``quartet-object-{class}_<ts>`` run via ModelRegistryPort, with
``quartet_metrics.json`` and per-model OOF parquets as artifacts.

The Black Box's final fit is what's stored as the run's primary
``model``; the other three are persisted as raw bytes inside
``artifacts``.
"""

from __future__ import annotations

import io
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import h3
import numpy as np
import polars as pl
from joblib import Parallel, delayed

from kadastra.adapters.catboost_quartet_model import CatBoostQuartetModel
from kadastra.adapters.ebm_quartet_model import EbmQuartetModel
from kadastra.adapters.grey_tree_quartet_model import GreyTreeQuartetModel
from kadastra.adapters.naive_linear_quartet_model import NaiveLinearQuartetModel
from kadastra.domain.asset_class import AssetClass
from kadastra.ml.metrics import regression_metrics
from kadastra.ml.object_feature_columns import select_object_feature_columns
from kadastra.ml.object_feature_matrix import build_object_feature_matrix
from kadastra.ml.quartet_checkpoint import QuartetCheckpointer, quartet_fingerprint
from kadastra.ml.quartet_metrics import (
    percentile_asymmetry,
    simplification_loss_pp,
    spearman_corr,
    wape,
)
from kadastra.ml.spatial_kfold import spatial_kfold_split
from kadastra.ml.train import CatBoostParams
from kadastra.ports.model_registry import ModelRegistryPort
from kadastra.ports.valuation_object_reader import ValuationObjectReaderPort

_TARGET_COLUMN = "synthetic_target_rub_per_m2"
_NAIVE_NUMERIC = ("lat", "lon", "area_m2", "levels", "flats", "year_built")
_NAIVE_CATEGORICAL = ("asset_class",)


def _log(msg: str) -> None:
    """Progress line, always flushed — a multi-hour run must not be a
    black box (stdout is block-buffered when redirected to a file)."""
    print(f"[train-quartet {datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


class TrainQuartet:
    def __init__(
        self,
        reader: ValuationObjectReaderPort,
        model_registry: ModelRegistryPort,
        *,
        catboost_params: CatBoostParams,
        ebm_max_bins: int,
        ebm_interactions: int,
        grey_tree_max_depth: int,
        n_splits: int,
        parent_resolution: int,
        parallel_folds: bool = False,
        skip_final_simplifier_fits: bool = False,
        checkpoint_dir: Path | None = None,
        resume: bool = True,
    ) -> None:
        self._reader = reader
        self._model_registry = model_registry
        self._catboost_params = catboost_params
        self._ebm_max_bins = ebm_max_bins
        self._ebm_interactions = ebm_interactions
        self._grey_tree_max_depth = grey_tree_max_depth
        self._n_splits = n_splits
        self._parent_resolution = parent_resolution
        # S1 (perf): when True, dispatch per-fold model fits via joblib
        # so n_splits folds train concurrently. Inner thread budget is
        # cpu_count // n_splits — pinning inner=1 made EBM run its
        # outer_bags (default 8) sequentially per fold, killing the
        # speedup; autoscale keeps total resident workers ≈ cpu.
        self._parallel_folds = parallel_folds
        # S2 (perf): when True, skip the EBM/Grey/Naive full-data refit
        # at the end of execute() — those *_model.pkl artifacts are not
        # consumed by any current code path (inspector reads OOFs only)
        # and dominate landplot wall time. CatBoost final fit is kept
        # because the model registry contract still requires a primary
        # CatBoostRegressor.
        self._skip_final_simplifier_fits = skip_final_simplifier_fits
        # Crash recovery: per-stage checkpoints under
        # ``checkpoint_dir / quartet-object-{class}``; a restarted run
        # with the same data+params fingerprint skips finished stages.
        self._checkpoint_dir = checkpoint_dir
        self._resume = resume

    def _run_stage(
        self,
        *,
        stage: str,
        args_list: list[tuple[Any, ...]],
        worker: Any,
        checkpointer: QuartetCheckpointer,
    ) -> list[dict[str, Any]]:
        """Run one pass over folds with per-fold crash recovery.

        Every fold result is checkpointed the moment it arrives
        (unordered streaming in parallel mode), so a kill loses at most
        the in-flight folds. A restarted run with the same fingerprint
        resumes finished folds instead of recomputing them.
        """
        results: list[dict[str, Any] | None] = [None] * len(args_list)
        missing: list[tuple[Any, ...]] = []
        for fold_id, args in enumerate(args_list):
            cached = checkpointer.load_stage(f"{stage}_fold_{fold_id}") if self._resume else None
            if cached is None:
                missing.append(args)
            else:
                results[fold_id] = cached
                _log(f"{stage} fold {fold_id + 1}/{len(args_list)}: resumed from checkpoint")
        if not missing:
            return cast("list[dict[str, Any]]", results)

        t_stage = time.perf_counter()
        done = len(args_list) - len(missing)
        if self._parallel_folds and len(missing) > 1:
            stream = Parallel(n_jobs=self._n_splits, backend="loky", return_as="generator_unordered")(
                delayed(worker)(*args) for args in missing
            )
            for r_any in stream:
                r = cast("dict[str, Any]", r_any)
                results[r["fold_id"]] = r
                checkpointer.save_stage(f"{stage}_fold_{r['fold_id']}", r)
                done += 1
                _log(
                    f"{stage} fold {r['fold_id'] + 1}/{len(args_list)} done in "
                    f"{r['elapsed_s'] / 60:.1f} min ({done}/{len(args_list)} complete)"
                )
        else:
            for args in missing:
                r = worker(*args)
                results[r["fold_id"]] = r
                checkpointer.save_stage(f"{stage}_fold_{r['fold_id']}", r)
                done += 1
                _log(
                    f"{stage} fold {r['fold_id'] + 1}/{len(args_list)} done in "
                    f"{r['elapsed_s'] / 60:.1f} min ({done}/{len(args_list)} complete)"
                )
        _log(f"{stage} complete: {len(missing)} folds computed in {(time.perf_counter() - t_stage) / 60:.1f} min")
        return cast("list[dict[str, Any]]", results)

    def execute(self, region_code: str, asset_class: AssetClass) -> str:
        t0 = time.perf_counter()
        df = self._reader.load(region_code, asset_class).drop_nulls(subset=[_TARGET_COLUMN])
        n = df.height
        y = df[_TARGET_COLUMN].to_numpy().astype(np.float64)
        _log(f"start class={asset_class.value} region={region_code} n={n}")

        # Full X: same matrix the per-class CatBoost training uses.
        full_numeric, full_categorical = select_object_feature_columns(df)
        full_feature_cols = full_numeric + full_categorical
        full_cat_idx = list(range(len(full_numeric), len(full_feature_cols)))
        X_full = build_object_feature_matrix(
            df,
            numeric_cols=full_numeric,
            categorical_cols=full_categorical,
        )

        # Naive X: raw fields only — measures the floor without
        # feature engineering.
        naive_numeric = [c for c in _NAIVE_NUMERIC if c in df.columns]
        naive_categorical = [c for c in _NAIVE_CATEGORICAL if c in df.columns]
        naive_feature_cols = naive_numeric + naive_categorical
        naive_cat_idx = list(range(len(naive_numeric), len(naive_feature_cols)))
        X_naive = build_object_feature_matrix(
            df,
            numeric_cols=naive_numeric,
            categorical_cols=naive_categorical,
        )

        # Spatial folds — one set, used by all four models.
        cell_resolution = max(self._parent_resolution + 1, 10)
        h3_indices = [
            h3.latlng_to_cell(float(lat), float(lon), cell_resolution)
            for lat, lon in zip(df["lat"].to_list(), df["lon"].to_list(), strict=True)
        ]
        folds = spatial_kfold_split(
            h3_indices,
            n_splits=self._n_splits,
            parent_resolution=self._parent_resolution,
            seed=self._catboost_params.seed,
        )

        checkpointer = QuartetCheckpointer(
            dir=(
                self._checkpoint_dir / f"quartet-object-{asset_class.value}"
                if self._checkpoint_dir is not None
                else None
            ),
            fingerprint=quartet_fingerprint(
                region_code=region_code,
                asset_class=asset_class.value,
                n_samples=n,
                y_bytes=y.tobytes(),
                full_feature_cols=full_feature_cols,
                naive_feature_cols=naive_feature_cols,
                n_splits=self._n_splits,
                parent_resolution=self._parent_resolution,
                catboost_params={
                    "iterations": self._catboost_params.iterations,
                    "learning_rate": self._catboost_params.learning_rate,
                    "depth": self._catboost_params.depth,
                    "seed": self._catboost_params.seed,
                },
                ebm_max_bins=self._ebm_max_bins,
                ebm_interactions=self._ebm_interactions,
                grey_tree_max_depth=self._grey_tree_max_depth,
            ),
        )
        if checkpointer.enabled:
            _log(f"checkpoints: {self._checkpoint_dir}/quartet-object-{asset_class.value} resume={self._resume}")

        # Pass 1: Black / White / Naive — per-fold fit + collect OOF.
        oof: dict[str, np.ndarray] = {
            "catboost": np.zeros(n, dtype=np.float64),
            "ebm": np.zeros(n, dtype=np.float64),
            "naive_linear": np.zeros(n, dtype=np.float64),
            "grey_tree": np.zeros(n, dtype=np.float64),
        }
        fold_ids = np.full(n, -1, dtype=np.int64)
        per_fold: dict[str, dict[str, list[float]]] = {
            m: {"mae": [], "rmse": [], "mape": []} for m in ("catboost", "ebm", "naive_linear", "grey_tree")
        }

        # When folds run in parallel, divide the available core budget
        # evenly across the n_splits fold workers — pinning inner=1
        # turned out to make EBM run its outer_bags (default 8)
        # *sequentially* per fold, killing most of the speedup. With
        # autoscale (cpu_count // n_splits) each fold worker gets a few
        # outer-bags-parallel slots, so total resident workers ≈ cpu
        # without oversubscription.
        if self._parallel_folds:
            cpu = os.cpu_count() or 8
            inner_threads = max(1, cpu // self._n_splits)
        else:
            inner_threads = None

        pass1_args = [
            (
                fold_id,
                np.array(train_idx_list, dtype=np.int64),
                np.array(val_idx_list, dtype=np.int64),
                X_full,
                X_naive,
                y,
                full_cat_idx,
                naive_cat_idx,
                self._catboost_params,
                self._ebm_max_bins,
                self._ebm_interactions,
                inner_threads,
            )
            for fold_id, (train_idx_list, val_idx_list) in enumerate(folds)
        ]
        pass1_results = self._run_stage(
            stage="pass1",
            args_list=pass1_args,
            worker=_fit_pass1_fold,
            checkpointer=checkpointer,
        )

        for r in pass1_results:
            val_idx = r["val_idx"]
            fold_ids[val_idx] = r["fold_id"]
            oof["catboost"][val_idx] = r["cb_pred"]
            oof["ebm"][val_idx] = r["ebm_pred"]
            oof["naive_linear"][val_idx] = r["nl_pred"]
            for model_name in ("catboost", "ebm", "naive_linear"):
                m = r["metrics"][model_name]
                per_fold[model_name]["mae"].append(m["mae"])
                per_fold[model_name]["rmse"].append(m["rmse"])
                per_fold[model_name]["mape"].append(m["mape"])

        # Pass 2: Grey Box on Black-OOF predictions. Per-fold so
        # Grey's val rows never see their own training target leak in.
        pass2_args = [
            (
                fold_id,
                np.array(train_idx_list, dtype=np.int64),
                np.array(val_idx_list, dtype=np.int64),
                X_full,
                oof["catboost"],
                y,
                full_cat_idx,
                self._grey_tree_max_depth,
                self._catboost_params.seed,
            )
            for fold_id, (train_idx_list, val_idx_list) in enumerate(folds)
        ]
        pass2_results = self._run_stage(
            stage="pass2",
            args_list=pass2_args,
            worker=_fit_pass2_grey_fold,
            checkpointer=checkpointer,
        )

        for r in pass2_results:
            val_idx = r["val_idx"]
            oof["grey_tree"][val_idx] = r["grey_pred"]
            m = r["metrics"]
            per_fold["grey_tree"]["mae"].append(m["mae"])
            per_fold["grey_tree"]["rmse"].append(m["rmse"])
            per_fold["grey_tree"]["mape"].append(m["mape"])

        # CatBoost final fit always runs — registry contract requires
        # a primary CatBoostRegressor as the run's model. Final fits are
        # checkpointed as serialized bytes: the artifact payload IS the
        # bytes, so a resumed run skips the refit entirely.
        cb_blob = checkpointer.load_stage("final_catboost") if self._resume else None
        if cb_blob is None:
            t_fit = time.perf_counter()
            bb_final = CatBoostQuartetModel(
                iterations=self._catboost_params.iterations,
                learning_rate=self._catboost_params.learning_rate,
                depth=self._catboost_params.depth,
                seed=self._catboost_params.seed,
            )
            bb_final.fit(X_full, y, cat_feature_indices=full_cat_idx or None)
            cb_blob = bb_final.serialize()
            checkpointer.save_stage("final_catboost", cb_blob)
            _log(f"final catboost fit: {(time.perf_counter() - t_fit) / 60:.1f} min")
        else:
            _log("final catboost: resumed from checkpoint")
        bb_final = CatBoostQuartetModel.deserialize(cast("bytes", cb_blob))

        # EBM (White Box) is always fit + saved — the inspector's
        # explanation endpoint loads ebm_model.pkl. Grey/Naive full-data
        # refits are not consumed (inspector reads OOFs only) and dominate
        # landplot wall time, so they stay skippable.
        ebm_blob = checkpointer.load_stage("final_ebm") if self._resume else None
        if ebm_blob is None:
            t_fit = time.perf_counter()
            wb_final = EbmQuartetModel(
                max_bins=self._ebm_max_bins,
                interactions=self._ebm_interactions,
            )
            wb_final.fit(X_full, y, cat_feature_indices=full_cat_idx or None)
            ebm_blob = wb_final.serialize()
            checkpointer.save_stage("final_ebm", ebm_blob)
            _log(f"final ebm fit: {(time.perf_counter() - t_fit) / 60:.1f} min")
        else:
            _log("final ebm: resumed from checkpoint")

        nl_blob: bytes | None = None
        grey_blob: bytes | None = None
        if not self._skip_final_simplifier_fits:
            nl_blob = checkpointer.load_stage("final_naive") if self._resume else None
            if nl_blob is None:
                t_fit = time.perf_counter()
                nl_final = NaiveLinearQuartetModel()
                nl_final.fit(X_naive, y, cat_feature_indices=naive_cat_idx or None)
                nl_blob = nl_final.serialize()
                checkpointer.save_stage("final_naive", nl_blob)
                _log(f"final naive fit: {(time.perf_counter() - t_fit) / 60:.1f} min")

            grey_blob = checkpointer.load_stage("final_grey") if self._resume else None
            if grey_blob is None:
                t_fit = time.perf_counter()
                grey_final = GreyTreeQuartetModel(
                    max_depth=self._grey_tree_max_depth,
                    seed=self._catboost_params.seed,
                )
                grey_final.fit(
                    X_full,
                    oof["catboost"],
                    cat_feature_indices=full_cat_idx or None,
                )
                grey_blob = grey_final.serialize()
                checkpointer.save_stage("final_grey", grey_blob)
                _log(f"final grey fit: {(time.perf_counter() - t_fit) / 60:.1f} min")

        # Aggregate metrics + Spearman + percentile asymmetry per model.
        # WAPE (ADR-0026) is the cross-class relative metric: stable on
        # the tiny ₽/м² denominator that breaks MAPE for landplot.
        models_payload: dict[str, dict[str, float]] = {}
        for model_name, fold_metrics in per_fold.items():
            agg = {
                "mean_mae": float(np.mean(fold_metrics["mae"])),
                "mean_rmse": float(np.mean(fold_metrics["rmse"])),
                "mean_mape": float(np.nanmean(fold_metrics["mape"])),
                "mean_spearman": spearman_corr(y, oof[model_name]),
                "wape": wape(y, oof[model_name]),
            }
            agg.update(percentile_asymmetry(y, oof[model_name]))
            models_payload[model_name] = agg

        # Grey fidelity to Black — R² on (catboost_oof, grey_oof).
        ss_res = float(np.sum((oof["grey_tree"] - oof["catboost"]) ** 2))
        ss_tot = float(np.sum((oof["catboost"] - np.mean(oof["catboost"])) ** 2))
        models_payload["grey_tree"]["fidelity_r2_to_catboost"] = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Loss on simplification (in percentage points).
        black_mape = models_payload["catboost"]["mean_mape"]
        ebm_mape = models_payload["ebm"]["mean_mape"]
        naive_mape = models_payload["naive_linear"]["mean_mape"]
        loss_payload = {
            "ebm_minus_catboost_mape_pp": simplification_loss_pp(black_mape, ebm_mape),
            "naive_minus_catboost_mape_pp": simplification_loss_pp(black_mape, naive_mape),
        }

        quartet_metrics = {
            "asset_class": asset_class.value,
            "n_samples": n,
            "n_splits": self._n_splits,
            "parent_resolution": self._parent_resolution,
            "models": models_payload,
            "loss_on_simplification": loss_payload,
        }

        artifacts: dict[str, bytes] = {
            "quartet_metrics.json": json.dumps(quartet_metrics, ensure_ascii=False, indent=2).encode("utf-8"),
            "catboost_oof_predictions.parquet": _build_oof_parquet(df, fold_ids, y, oof["catboost"]),
            "ebm_oof_predictions.parquet": _build_oof_parquet(df, fold_ids, y, oof["ebm"]),
            "grey_tree_oof_predictions.parquet": _build_oof_parquet(df, fold_ids, y, oof["grey_tree"]),
            "naive_linear_oof_predictions.parquet": _build_oof_parquet(df, fold_ids, y, oof["naive_linear"]),
        }
        artifacts["ebm_model.pkl"] = cast("bytes", ebm_blob)
        if grey_blob is not None:
            artifacts["grey_tree_model.pkl"] = grey_blob
        if nl_blob is not None:
            artifacts["naive_linear_model.pkl"] = nl_blob

        params_payload = {
            "asset_class": asset_class.value,
            "n_samples": n,
            "n_splits": self._n_splits,
            "parent_resolution": self._parent_resolution,
            "feature_columns_full": full_feature_cols,
            "feature_columns_naive": naive_feature_cols,
            "catboost_params": {
                "iterations": self._catboost_params.iterations,
                "learning_rate": self._catboost_params.learning_rate,
                "depth": self._catboost_params.depth,
                "seed": self._catboost_params.seed,
            },
            "ebm_max_bins": self._ebm_max_bins,
            "ebm_interactions": self._ebm_interactions,
            "grey_tree_max_depth": self._grey_tree_max_depth,
        }

        flat_metrics = {
            f"{model_name}__{key}": value
            for model_name, model_metrics in models_payload.items()
            for key, value in model_metrics.items()
        }
        flat_metrics.update(loss_payload)

        # The CatBoost final fit is the run's primary model; other
        # three live in artifacts. ModelRegistryPort accepts a
        # CatBoostRegressor here, which our adapter exposes via the
        # underlying ``_model``.
        run_id = self._model_registry.log_run(
            run_name=f"quartet-object-{asset_class.value}",
            params=params_payload,
            metrics=flat_metrics,
            model=bb_final.unwrap(),
            artifacts=artifacts,
        )
        checkpointer.discard()
        _log(f"done class={asset_class.value} run_id={run_id} total={(time.perf_counter() - t0) / 3600:.2f} h")
        return run_id


def _fit_pass1_fold(
    fold_id: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    X_full: np.ndarray,
    X_naive: np.ndarray,
    y: np.ndarray,
    full_cat_idx: list[int],
    naive_cat_idx: list[int],
    catboost_params: CatBoostParams,
    ebm_max_bins: int,
    ebm_interactions: int,
    inner_threads: int | None,
) -> dict[str, Any]:
    """Train Black/White/Naive on one fold and return per-fold OOF
    predictions + metrics. Top-level function so joblib can pickle it
    when dispatched across processes."""
    t_fold = time.perf_counter()

    t = time.perf_counter()
    bb = CatBoostQuartetModel(
        iterations=catboost_params.iterations,
        learning_rate=catboost_params.learning_rate,
        depth=catboost_params.depth,
        seed=catboost_params.seed,
        thread_count=inner_threads,
    )
    bb.fit(X_full[train_idx], y[train_idx], cat_feature_indices=full_cat_idx or None)
    cb_pred = bb.predict(X_full[val_idx])
    cb_metrics = regression_metrics(y[val_idx], cb_pred)
    cb_s = time.perf_counter() - t
    _log(f"pass1 fold {fold_id + 1}: catboost fit {cb_s / 60:.1f} min")

    t = time.perf_counter()
    wb = EbmQuartetModel(
        max_bins=ebm_max_bins,
        interactions=ebm_interactions,
        n_jobs=inner_threads,
    )
    wb.fit(X_full[train_idx], y[train_idx], cat_feature_indices=full_cat_idx or None)
    ebm_pred = wb.predict(X_full[val_idx])
    ebm_metrics = regression_metrics(y[val_idx], ebm_pred)
    ebm_s = time.perf_counter() - t
    _log(f"pass1 fold {fold_id + 1}: ebm fit {ebm_s / 60:.1f} min")

    t = time.perf_counter()
    nl = NaiveLinearQuartetModel()
    nl.fit(
        X_naive[train_idx],
        y[train_idx],
        cat_feature_indices=naive_cat_idx or None,
    )
    nl_pred = nl.predict(X_naive[val_idx])
    nl_metrics = regression_metrics(y[val_idx], nl_pred)
    nl_s = time.perf_counter() - t
    _log(f"pass1 fold {fold_id + 1}: naive fit {nl_s / 60:.1f} min")

    return {
        "fold_id": fold_id,
        "val_idx": val_idx,
        "cb_pred": cb_pred,
        "ebm_pred": ebm_pred,
        "nl_pred": nl_pred,
        "elapsed_s": time.perf_counter() - t_fold,
        "model_timings_s": {"catboost": cb_s, "ebm": ebm_s, "naive_linear": nl_s},
        "metrics": {
            "catboost": cb_metrics,
            "ebm": ebm_metrics,
            "naive_linear": nl_metrics,
        },
    }


def _fit_pass2_grey_fold(
    fold_id: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    X_full: np.ndarray,
    catboost_oof: np.ndarray,
    y: np.ndarray,
    full_cat_idx: list[int],
    grey_tree_max_depth: int,
    seed: int,
) -> dict[str, Any]:
    """Grey Box fold: fit on Black-OOF predictions for train rows,
    predict on val. Top-level function so joblib can pickle it."""
    t_fold = time.perf_counter()
    grey = GreyTreeQuartetModel(max_depth=grey_tree_max_depth, seed=seed)
    grey.fit(
        X_full[train_idx],
        catboost_oof[train_idx],
        cat_feature_indices=full_cat_idx or None,
    )
    grey_pred = grey.predict(X_full[val_idx])
    return {
        "fold_id": fold_id,
        "val_idx": val_idx,
        "grey_pred": grey_pred,
        "elapsed_s": time.perf_counter() - t_fold,
        # Grey fold metrics computed against y_true so they're
        # comparable to the rest; fidelity to Black is reported
        # separately at the aggregate level.
        "metrics": regression_metrics(y[val_idx], grey_pred),
    }


def _build_oof_parquet(
    df: pl.DataFrame,
    fold_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred_oof: np.ndarray,
) -> bytes:
    out = pl.DataFrame(
        {
            "object_id": df["object_id"].to_list(),
            "lat": df["lat"].to_list(),
            "lon": df["lon"].to_list(),
            "fold_id": fold_ids.tolist(),
            "y_true": y_true.tolist(),
            "y_pred_oof": y_pred_oof.tolist(),
        },
        schema={
            "object_id": pl.Utf8,
            "lat": pl.Float64,
            "lon": pl.Float64,
            "fold_id": pl.Int64,
            "y_true": pl.Float64,
            "y_pred_oof": pl.Float64,
        },
    ).sort("object_id")
    buf = io.BytesIO()
    out.write_parquet(buf)
    return buf.getvalue()
