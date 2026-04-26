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
from typing import Any

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
from kadastra.ml.quartet_metrics import (
    percentile_asymmetry,
    simplification_loss_pp,
    spearman_corr,
)
from kadastra.ml.spatial_kfold import spatial_kfold_split
from kadastra.ml.train import CatBoostParams
from kadastra.ports.model_registry import ModelRegistryPort
from kadastra.ports.valuation_object_reader import ValuationObjectReaderPort

_TARGET_COLUMN = "synthetic_target_rub_per_m2"
_NAIVE_NUMERIC = ("lat", "lon", "area_m2", "levels", "flats", "year_built")
_NAIVE_CATEGORICAL = ("asset_class",)


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
        # so n_splits folds train concurrently. Inner thread pools are
        # narrowed to 1 to avoid n_splits × outer_bags worker explosion.
        # Logic for this flag is implemented in a follow-up commit.
        self._parallel_folds = parallel_folds
        # S2 (perf): when True, skip the EBM/Grey/Naive full-data refit
        # at the end of execute() — those *_model.pkl artifacts are not
        # consumed by any current code path (inspector reads OOFs only)
        # and dominate landplot wall time. CatBoost final fit is kept
        # because the model registry contract still requires a primary
        # CatBoostRegressor. Logic implemented in a follow-up commit.
        self._skip_final_simplifier_fits = skip_final_simplifier_fits

    def execute(self, region_code: str, asset_class: AssetClass) -> str:
        df = self._reader.load(region_code, asset_class).drop_nulls(
            subset=[_TARGET_COLUMN]
        )
        n = df.height
        y = df[_TARGET_COLUMN].to_numpy().astype(np.float64)

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
            for lat, lon in zip(
                df["lat"].to_list(), df["lon"].to_list(), strict=True
            )
        ]
        folds = spatial_kfold_split(
            h3_indices,
            n_splits=self._n_splits,
            parent_resolution=self._parent_resolution,
            seed=self._catboost_params.seed,
        )

        # Pass 1: Black / White / Naive — per-fold fit + collect OOF.
        oof: dict[str, np.ndarray] = {
            "catboost": np.zeros(n, dtype=np.float64),
            "ebm": np.zeros(n, dtype=np.float64),
            "naive_linear": np.zeros(n, dtype=np.float64),
            "grey_tree": np.zeros(n, dtype=np.float64),
        }
        fold_ids = np.full(n, -1, dtype=np.int64)
        per_fold: dict[str, dict[str, list[float]]] = {
            m: {"mae": [], "rmse": [], "mape": []}
            for m in ("catboost", "ebm", "naive_linear", "grey_tree")
        }

        # When folds run in parallel, inner thread pools are pinned to 1
        # so n_splits × default-all-cores doesn't oversubscribe the box.
        inner_threads = 1 if self._parallel_folds else None

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
        if self._parallel_folds:
            pass1_results = Parallel(
                n_jobs=self._n_splits, backend="loky"
            )(delayed(_fit_pass1_fold)(*args) for args in pass1_args)
        else:
            pass1_results = [_fit_pass1_fold(*args) for args in pass1_args]

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
                np.array(train_idx_list, dtype=np.int64),
                np.array(val_idx_list, dtype=np.int64),
                X_full,
                oof["catboost"],
                y,
                full_cat_idx,
                self._grey_tree_max_depth,
                self._catboost_params.seed,
            )
            for train_idx_list, val_idx_list in folds
        ]
        if self._parallel_folds:
            pass2_results = Parallel(
                n_jobs=self._n_splits, backend="loky"
            )(delayed(_fit_pass2_grey_fold)(*args) for args in pass2_args)
        else:
            pass2_results = [
                _fit_pass2_grey_fold(*args) for args in pass2_args
            ]

        for r in pass2_results:
            val_idx = r["val_idx"]
            oof["grey_tree"][val_idx] = r["grey_pred"]
            m = r["metrics"]
            per_fold["grey_tree"]["mae"].append(m["mae"])
            per_fold["grey_tree"]["rmse"].append(m["rmse"])
            per_fold["grey_tree"]["mape"].append(m["mape"])

        # CatBoost final fit always runs — registry contract requires
        # a primary CatBoostRegressor as the run's model.
        bb_final = CatBoostQuartetModel(
            iterations=self._catboost_params.iterations,
            learning_rate=self._catboost_params.learning_rate,
            depth=self._catboost_params.depth,
            seed=self._catboost_params.seed,
        )
        bb_final.fit(X_full, y, cat_feature_indices=full_cat_idx or None)

        # The three simplifiers' full-data refits are not consumed by
        # any current code path (inspector reads OOFs only). On
        # landplot they dominate wall time, so they're skippable.
        wb_final: EbmQuartetModel | None = None
        nl_final: NaiveLinearQuartetModel | None = None
        grey_final: GreyTreeQuartetModel | None = None
        if not self._skip_final_simplifier_fits:
            wb_final = EbmQuartetModel(
                max_bins=self._ebm_max_bins,
                interactions=self._ebm_interactions,
            )
            wb_final.fit(X_full, y, cat_feature_indices=full_cat_idx or None)

            nl_final = NaiveLinearQuartetModel()
            nl_final.fit(
                X_naive, y, cat_feature_indices=naive_cat_idx or None
            )

            grey_final = GreyTreeQuartetModel(
                max_depth=self._grey_tree_max_depth,
                seed=self._catboost_params.seed,
            )
            grey_final.fit(
                X_full,
                oof["catboost"],
                cat_feature_indices=full_cat_idx or None,
            )

        # Aggregate metrics + Spearman + percentile asymmetry per model.
        models_payload: dict[str, dict[str, float]] = {}
        for model_name, fold_metrics in per_fold.items():
            agg = {
                "mean_mae": float(np.mean(fold_metrics["mae"])),
                "mean_rmse": float(np.mean(fold_metrics["rmse"])),
                "mean_mape": float(np.nanmean(fold_metrics["mape"])),
                "mean_spearman": spearman_corr(y, oof[model_name]),
            }
            agg.update(percentile_asymmetry(y, oof[model_name]))
            models_payload[model_name] = agg

        # Grey fidelity to Black — R² on (catboost_oof, grey_oof).
        ss_res = float(np.sum((oof["grey_tree"] - oof["catboost"]) ** 2))
        ss_tot = float(
            np.sum((oof["catboost"] - np.mean(oof["catboost"])) ** 2)
        )
        models_payload["grey_tree"]["fidelity_r2_to_catboost"] = (
            float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        )

        # Loss on simplification (in percentage points).
        black_mape = models_payload["catboost"]["mean_mape"]
        ebm_mape = models_payload["ebm"]["mean_mape"]
        naive_mape = models_payload["naive_linear"]["mean_mape"]
        loss_payload = {
            "ebm_minus_catboost_mape_pp": simplification_loss_pp(
                black_mape, ebm_mape
            ),
            "naive_minus_catboost_mape_pp": simplification_loss_pp(
                black_mape, naive_mape
            ),
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
            "quartet_metrics.json": json.dumps(
                quartet_metrics, ensure_ascii=False, indent=2
            ).encode("utf-8"),
            "catboost_oof_predictions.parquet": _build_oof_parquet(
                df, fold_ids, y, oof["catboost"]
            ),
            "ebm_oof_predictions.parquet": _build_oof_parquet(
                df, fold_ids, y, oof["ebm"]
            ),
            "grey_tree_oof_predictions.parquet": _build_oof_parquet(
                df, fold_ids, y, oof["grey_tree"]
            ),
            "naive_linear_oof_predictions.parquet": _build_oof_parquet(
                df, fold_ids, y, oof["naive_linear"]
            ),
        }
        if wb_final is not None:
            artifacts["ebm_model.pkl"] = wb_final.serialize()
        if grey_final is not None:
            artifacts["grey_tree_model.pkl"] = grey_final.serialize()
        if nl_final is not None:
            artifacts["naive_linear_model.pkl"] = nl_final.serialize()

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
        return self._model_registry.log_run(
            run_name=f"quartet-object-{asset_class.value}",
            params=params_payload,
            metrics=flat_metrics,
            model=bb_final.unwrap(),
            artifacts=artifacts,
        )


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

    wb = EbmQuartetModel(
        max_bins=ebm_max_bins,
        interactions=ebm_interactions,
        n_jobs=inner_threads,
    )
    wb.fit(X_full[train_idx], y[train_idx], cat_feature_indices=full_cat_idx or None)
    ebm_pred = wb.predict(X_full[val_idx])
    ebm_metrics = regression_metrics(y[val_idx], ebm_pred)

    nl = NaiveLinearQuartetModel()
    nl.fit(
        X_naive[train_idx],
        y[train_idx],
        cat_feature_indices=naive_cat_idx or None,
    )
    nl_pred = nl.predict(X_naive[val_idx])
    nl_metrics = regression_metrics(y[val_idx], nl_pred)

    return {
        "fold_id": fold_id,
        "val_idx": val_idx,
        "cb_pred": cb_pred,
        "ebm_pred": ebm_pred,
        "nl_pred": nl_pred,
        "metrics": {
            "catboost": cb_metrics,
            "ebm": ebm_metrics,
            "naive_linear": nl_metrics,
        },
    }


def _fit_pass2_grey_fold(
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
    grey = GreyTreeQuartetModel(max_depth=grey_tree_max_depth, seed=seed)
    grey.fit(
        X_full[train_idx],
        catboost_oof[train_idx],
        cat_feature_indices=full_cat_idx or None,
    )
    grey_pred = grey.predict(X_full[val_idx])
    return {
        "val_idx": val_idx,
        "grey_pred": grey_pred,
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
