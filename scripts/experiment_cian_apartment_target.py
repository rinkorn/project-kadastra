"""EXPERIMENTAL (ADR-0031, variant б): CIAN listings-as-target for apartment.

Not part of the production train path. Trains the Black/White/Grey/Naive
quartet twice on the SAME spatial-CV folds over the CIAN-matched subset:

- baseline: target = ``synthetic_target_rub_per_m2`` (ЕГРН cost_index) of the
  matched object;
- cian_ask: target = ``ask_rub_per_m2`` of the matched listing (market ask).

Features are the matched object's features only (same column selection as
TrainQuartet) — listing price/area fields are never used as features, so no
target leak. Fold construction mirrors TrainQuartet exactly
(h3 res = max(parent+1, 10), spatial_kfold_split, same seed/params).

MAE/RMSE are NOT comparable across the two targets (ask ≈ 2.21 × cost_index);
compare MAPE/WAPE/Spearman only.

Usage:
    uv run python scripts/experiment_cian_apartment_target.py
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h3
import numpy as np
import polars as pl

from kadastra.config import Settings
from kadastra.ml.object_feature_columns import select_object_feature_columns
from kadastra.ml.object_feature_matrix import build_object_feature_matrix
from kadastra.ml.quartet_metrics import (
    percentile_asymmetry,
    simplification_loss_pp,
    spearman_corr,
    wape,
)
from kadastra.ml.spatial_kfold import spatial_kfold_split
from kadastra.ml.train import CatBoostParams
from kadastra.usecases.train_quartet import _fit_pass1_fold, _fit_pass2_grey_fold

_OBJECTS_PARQUET = Path("data/gold/valuation_objects/region=RU-KAZAN-AGG/asset_class=apartment/data.parquet")
_MATCHED_PARQUET = Path("data/silver/listings_target/region=RU-KAZAN-AGG/asset_class=apartment/matched.parquet")
_TARGET_COLUMN = "synthetic_target_rub_per_m2"
_NAIVE_NUMERIC = ("lat", "lon", "area_m2", "levels", "flats", "year_built")
_NAIVE_CATEGORICAL = ("asset_class",)
_OUT_JSON = Path(".tmp_research/cian_apartment_target_experiment.json")


def _run_quartet_cv(
    y: np.ndarray,
    X_full: np.ndarray,
    X_naive: np.ndarray,
    full_cat_idx: list[int],
    naive_cat_idx: list[int],
    folds: list[tuple[list[int], list[int]]],
    catboost_params: CatBoostParams,
    settings: Settings,
) -> tuple[dict[str, dict[str, float]], dict[str, list[dict[str, float]]], dict[str, np.ndarray]]:
    """Mirror of TrainQuartet passes 1+2 for one target: per-fold fit,
    OOF, aggregate metrics. Returns (aggregate, per_fold, oof)."""
    n = y.shape[0]
    oof = {m: np.zeros(n, dtype=np.float64) for m in ("catboost", "ebm", "naive_linear", "grey_tree")}
    per_fold: dict[str, list[dict[str, float]]] = {m: [] for m in ("catboost", "ebm", "naive_linear", "grey_tree")}

    for fold_id, (train_idx_list, val_idx_list) in enumerate(folds):
        r = _fit_pass1_fold(
            fold_id,
            np.array(train_idx_list, dtype=np.int64),
            np.array(val_idx_list, dtype=np.int64),
            X_full,
            X_naive,
            y,
            full_cat_idx,
            naive_cat_idx,
            catboost_params,
            settings.ebm_max_bins,
            settings.ebm_interactions,
            None,
        )
        val_idx = r["val_idx"]
        oof["catboost"][val_idx] = r["cb_pred"]
        oof["ebm"][val_idx] = r["ebm_pred"]
        oof["naive_linear"][val_idx] = r["nl_pred"]
        for model_name in ("catboost", "ebm", "naive_linear"):
            per_fold[model_name].append(r["metrics"][model_name])

    for train_idx_list, val_idx_list in folds:
        r = _fit_pass2_grey_fold(
            np.array(train_idx_list, dtype=np.int64),
            np.array(val_idx_list, dtype=np.int64),
            X_full,
            oof["catboost"],
            y,
            full_cat_idx,
            settings.grey_tree_max_depth,
            catboost_params.seed,
        )
        val_idx = r["val_idx"]
        oof["grey_tree"][val_idx] = r["grey_pred"]
        per_fold["grey_tree"].append(r["metrics"])

    aggregate: dict[str, dict[str, float]] = {}
    for model_name, fold_metrics in per_fold.items():
        agg = {
            "mean_mae": float(np.mean([m["mae"] for m in fold_metrics])),
            "mean_rmse": float(np.mean([m["rmse"] for m in fold_metrics])),
            "mean_mape": float(np.nanmean([m["mape"] for m in fold_metrics])),
            "mean_spearman": spearman_corr(y, oof[model_name]),
            "wape": wape(y, oof[model_name]),
        }
        agg.update(percentile_asymmetry(y, oof[model_name]))
        aggregate[model_name] = agg

    ss_res = float(np.sum((oof["grey_tree"] - oof["catboost"]) ** 2))
    ss_tot = float(np.sum((oof["catboost"] - np.mean(oof["catboost"])) ** 2))
    aggregate["grey_tree"]["fidelity_r2_to_catboost"] = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return aggregate, per_fold, oof


def main() -> None:
    settings = Settings()
    catboost_params = CatBoostParams(
        iterations=settings.catboost_iterations,
        learning_rate=settings.catboost_learning_rate,
        depth=settings.catboost_depth,
        seed=settings.catboost_seed,
    )

    objects = pl.read_parquet(_OBJECTS_PARQUET).drop_nulls(subset=[_TARGET_COLUMN])
    matched = pl.read_parquet(_MATCHED_PARQUET)
    joined = matched.select("listing_id", "matched_object_id", "ask_rub_per_m2").join(
        objects, left_on="matched_object_id", right_on="object_id", how="inner"
    )
    n_listings = matched.height
    n = joined.height
    if n == 0:
        raise SystemExit("join produced zero rows — check matched_object_id vs object_id")

    y_ask = joined["ask_rub_per_m2"].to_numpy().astype(np.float64)
    y_cost = joined[_TARGET_COLUMN].to_numpy().astype(np.float64)
    ratio = y_ask / y_cost
    print(
        f"matched listings={n_listings} joined={n} "
        f"unique_objects={joined['matched_object_id'].n_unique()} "
        f"ask/cost_index median={float(np.median(ratio)):.3f}",
        flush=True,
    )

    # Features: object columns only — listing fields never enter the matrix.
    # polars drops the right join key (object_id) — it's a non-feature
    # column anyway, so exclude it from the feature frame.
    df_feat = joined.select([c for c in objects.columns if c != "object_id"])
    full_numeric, full_categorical = select_object_feature_columns(df_feat)
    full_feature_cols = full_numeric + full_categorical
    full_cat_idx = list(range(len(full_numeric), len(full_feature_cols)))
    X_full = build_object_feature_matrix(df_feat, numeric_cols=full_numeric, categorical_cols=full_categorical)
    naive_numeric = [c for c in _NAIVE_NUMERIC if c in df_feat.columns]
    naive_categorical = [c for c in _NAIVE_CATEGORICAL if c in df_feat.columns]
    naive_feature_cols = naive_numeric + naive_categorical
    naive_cat_idx = list(range(len(naive_numeric), len(naive_feature_cols)))
    X_naive = build_object_feature_matrix(df_feat, numeric_cols=naive_numeric, categorical_cols=naive_categorical)

    # Same fold construction as TrainQuartet.
    cell_resolution = max(settings.train_parent_resolution + 1, 10)
    h3_indices = [
        h3.latlng_to_cell(float(lat), float(lon), cell_resolution)
        for lat, lon in zip(df_feat["lat"].to_list(), df_feat["lon"].to_list(), strict=True)
    ]
    folds = spatial_kfold_split(
        h3_indices,
        n_splits=settings.train_n_splits,
        parent_resolution=settings.train_parent_resolution,
        seed=catboost_params.seed,
    )
    print(
        f"folds: n_splits={settings.train_n_splits} "
        f"parent_res={settings.train_parent_resolution} "
        f"val_sizes={[len(v) for _, v in folds]}",
        flush=True,
    )

    results: dict[str, Any] = {}
    for target_name, y in (("cost_index", y_cost), ("cian_ask", y_ask)):
        print(f"--- target={target_name} ---", flush=True)
        aggregate, per_fold, _oof = _run_quartet_cv(
            y, X_full, X_naive, full_cat_idx, naive_cat_idx, folds, catboost_params, settings
        )
        black_mape = aggregate["catboost"]["mean_mape"]
        loss = {
            "ebm_minus_catboost_mape_pp": simplification_loss_pp(black_mape, aggregate["ebm"]["mean_mape"]),
            "naive_minus_catboost_mape_pp": simplification_loss_pp(black_mape, aggregate["naive_linear"]["mean_mape"]),
        }
        results[target_name] = {
            "aggregate": aggregate,
            "per_fold": per_fold,
            "loss_on_simplification": loss,
        }
        for model_name, agg in aggregate.items():
            print(
                f"  {model_name:13s} mae={agg['mean_mae']:10.1f} "
                f"rmse={agg['mean_rmse']:10.1f} mape={agg['mean_mape']:.4f} "
                f"wape={agg['wape']:.4f} spearman={agg['mean_spearman']:.4f}",
                flush=True,
            )

    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "n_listings_matched": n_listings,
        "n_rows_joined": n,
        "n_unique_objects": int(joined["matched_object_id"].n_unique()),
        "ask_over_cost_index_median": float(np.median(ratio)),
        "n_splits": settings.train_n_splits,
        "parent_resolution": settings.train_parent_resolution,
        "results": results,
    }
    _OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    _OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"saved: {_OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
