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

import h3
import numpy as np
import polars as pl

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

        for fold_id, (train_idx_list, val_idx_list) in enumerate(folds):
            train_idx = np.array(train_idx_list, dtype=np.int64)
            val_idx = np.array(val_idx_list, dtype=np.int64)
            fold_ids[val_idx] = fold_id

            # Black Box (CatBoost).
            bb = CatBoostQuartetModel(
                iterations=self._catboost_params.iterations,
                learning_rate=self._catboost_params.learning_rate,
                depth=self._catboost_params.depth,
                seed=self._catboost_params.seed,
            )
            bb.fit(
                X_full[train_idx],
                y[train_idx],
                cat_feature_indices=full_cat_idx or None,
            )
            bb_pred = bb.predict(X_full[val_idx])
            oof["catboost"][val_idx] = bb_pred
            _record_fold_metrics(per_fold["catboost"], y[val_idx], bb_pred)

            # White Box (EBM).
            wb = EbmQuartetModel(
                max_bins=self._ebm_max_bins,
                interactions=self._ebm_interactions,
            )
            wb.fit(
                X_full[train_idx],
                y[train_idx],
                cat_feature_indices=full_cat_idx or None,
            )
            wb_pred = wb.predict(X_full[val_idx])
            oof["ebm"][val_idx] = wb_pred
            _record_fold_metrics(per_fold["ebm"], y[val_idx], wb_pred)

            # Naive Linear.
            nl = NaiveLinearQuartetModel()
            nl.fit(
                X_naive[train_idx],
                y[train_idx],
                cat_feature_indices=naive_cat_idx or None,
            )
            nl_pred = nl.predict(X_naive[val_idx])
            oof["naive_linear"][val_idx] = nl_pred
            _record_fold_metrics(per_fold["naive_linear"], y[val_idx], nl_pred)

        # Pass 2: Grey Box on Black-OOF predictions. Per-fold so
        # Grey's val rows never see their own training target leak in.
        for train_idx_list, val_idx_list in folds:
            train_idx = np.array(train_idx_list, dtype=np.int64)
            val_idx = np.array(val_idx_list, dtype=np.int64)

            grey = GreyTreeQuartetModel(
                max_depth=self._grey_tree_max_depth,
                seed=self._catboost_params.seed,
            )
            grey.fit(
                X_full[train_idx],
                oof["catboost"][train_idx],
                cat_feature_indices=full_cat_idx or None,
            )
            grey_pred = grey.predict(X_full[val_idx])
            oof["grey_tree"][val_idx] = grey_pred
            # Grey fold metrics computed against y_true so they're
            # comparable to the rest; fidelity to Black is reported
            # separately below.
            _record_fold_metrics(per_fold["grey_tree"], y[val_idx], grey_pred)

        # Final fits on the full data — only the CatBoost one is
        # passed as the "model" of the run; the other three are
        # serialized into artifacts.
        bb_final = CatBoostQuartetModel(
            iterations=self._catboost_params.iterations,
            learning_rate=self._catboost_params.learning_rate,
            depth=self._catboost_params.depth,
            seed=self._catboost_params.seed,
        )
        bb_final.fit(X_full, y, cat_feature_indices=full_cat_idx or None)

        wb_final = EbmQuartetModel(
            max_bins=self._ebm_max_bins,
            interactions=self._ebm_interactions,
        )
        wb_final.fit(X_full, y, cat_feature_indices=full_cat_idx or None)

        nl_final = NaiveLinearQuartetModel()
        nl_final.fit(X_naive, y, cat_feature_indices=naive_cat_idx or None)

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
            "ebm_model.pkl": wb_final.serialize(),
            "grey_tree_model.pkl": grey_final.serialize(),
            "naive_linear_model.pkl": nl_final.serialize(),
        }

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


def _record_fold_metrics(
    bucket: dict[str, list[float]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    m = regression_metrics(y_true, y_pred)
    bucket["mae"].append(m["mae"])
    bucket["rmse"].append(m["rmse"])
    bucket["mape"].append(m["mape"])


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
