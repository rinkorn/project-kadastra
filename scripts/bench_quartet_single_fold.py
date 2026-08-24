"""Bench: wall-time per quartet model on ONE spatial fold (landplot).

Replicates TrainQuartet's data path exactly (same parquet, same feature
selection, same matrix builder, same spatial_kfold_split parameters),
but trains a single fold sequentially with perf_counter timings printed
with flush=True. Thread budget is pinned low (CatBoost thread_count=2,
EBM n_jobs=1, no joblib) so the bench can run next to the production
training without starving it.

Usage:
    nice -n 15 uv run python scripts/bench_quartet_single_fold.py
    nice -n 15 uv run python scripts/bench_quartet_single_fold.py --sentinel
    nice -n 15 uv run python scripts/bench_quartet_single_fold.py --sample 20000
"""

from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import h3
import numpy as np
import polars as pl

from kadastra.adapters.catboost_quartet_model import CatBoostQuartetModel
from kadastra.adapters.ebm_quartet_model import EbmQuartetModel
from kadastra.adapters.grey_tree_quartet_model import GreyTreeQuartetModel
from kadastra.adapters.naive_linear_quartet_model import NaiveLinearQuartetModel
from kadastra.ml.object_feature_columns import select_object_feature_columns
from kadastra.ml.object_feature_matrix import build_object_feature_matrix
from kadastra.ml.spatial_kfold import spatial_kfold_split

_PARQUET = "data/gold/valuation_objects/region=RU-KAZAN-AGG/asset_class=landplot/data.parquet"
_TARGET_COLUMN = "synthetic_target_rub_per_m2"
_NAIVE_NUMERIC = ("lat", "lon", "area_m2", "levels", "flats", "year_built")
_NAIVE_CATEGORICAL = ("asset_class",)

# Same values as Settings defaults (config.py).
_CATBOOST_ITERATIONS = 500
_CATBOOST_LR = 0.05
_CATBOOST_DEPTH = 6
_SEED = 42
_EBM_MAX_BINS = 256
_EBM_INTERACTIONS = 10
_GREY_MAX_DEPTH = 10
_N_SPLITS = 5
_PARENT_RES = 7

_GRAPH_COLUMNS = (
    "dist_metro_m",
    "dist_entrance_m",
    "walk_dist_to_school_m",
    "walk_dist_to_kindergarten_m",
    "walk_dist_to_clinic_m",
    "walk_dist_to_hospital_m",
    "walk_dist_to_pharmacy_m",
    "walk_dist_to_supermarket_m",
    "walk_dist_to_cafe_m",
    "walk_dist_to_restaurant_m",
    "walk_dist_to_bus_stop_m",
    "walk_dist_to_tram_stop_m",
    "walk_dist_to_railway_station_m",
)


def _t() -> float:
    return time.perf_counter()


def _report(label: str, start: float) -> None:
    print(f"[bench] {label}: {_t() - start:.1f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--sentinel",
        action="store_true",
        help="Replace NaN/None with 1e9 in graph-distance columns "
        "(simulates the pre-recalc feature state for the A/B).",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Subsample TRAIN rows (deterministic) for a quick smoke run.",
    )
    args = parser.parse_args()

    t0 = _t()
    df = pl.read_parquet(_PARQUET).drop_nulls(subset=[_TARGET_COLUMN])
    _report(f"load parquet ({df.height} rows)", t0)

    if args.sentinel:
        t = _t()
        replaced = 0
        for c in _GRAPH_COLUMNS:
            if c not in df.columns:
                continue
            n_bad = int(df[c].is_null().sum() + df[c].is_nan().sum())
            if n_bad:
                df = df.with_columns(
                    pl.when(pl.col(c).is_null() | pl.col(c).is_nan()).then(pl.lit(1e9)).otherwise(pl.col(c)).alias(c)
                )
                replaced += n_bad
        _report(f"sentinel fill (replaced {replaced} cells)", t)

    y = df[_TARGET_COLUMN].to_numpy().astype(np.float64)

    t = _t()
    full_numeric, full_categorical = select_object_feature_columns(df)
    full_feature_cols = full_numeric + full_categorical
    full_cat_idx = list(range(len(full_numeric), len(full_feature_cols)))
    print(
        f"[bench] features: {len(full_numeric)} numeric + {len(full_categorical)} categorical",
        flush=True,
    )
    X_full = build_object_feature_matrix(
        df,
        numeric_cols=full_numeric,
        categorical_cols=full_categorical,
    )
    _report(f"build X_full {X_full.shape}", t)

    t = _t()
    naive_numeric = [c for c in _NAIVE_NUMERIC if c in df.columns]
    naive_categorical = [c for c in _NAIVE_CATEGORICAL if c in df.columns]
    naive_feature_cols = naive_numeric + naive_categorical
    naive_cat_idx = list(range(len(naive_numeric), len(naive_feature_cols)))
    X_naive = build_object_feature_matrix(
        df,
        numeric_cols=naive_numeric,
        categorical_cols=naive_categorical,
    )
    _report(f"build X_naive {X_naive.shape}", t)

    t = _t()
    cell_resolution = max(_PARENT_RES + 1, 10)
    h3_indices = [
        h3.latlng_to_cell(float(lat), float(lon), cell_resolution)
        for lat, lon in zip(df["lat"].to_list(), df["lon"].to_list(), strict=True)
    ]
    folds = spatial_kfold_split(
        h3_indices,
        n_splits=_N_SPLITS,
        parent_resolution=_PARENT_RES,
        seed=_SEED,
    )
    _report("spatial folds", t)

    fold_id = args.fold
    train_idx = np.array(folds[fold_id][0], dtype=np.int64)
    val_idx = np.array(folds[fold_id][1], dtype=np.int64)
    if args.sample is not None and args.sample < len(train_idx):
        rng = np.random.default_rng(_SEED)
        train_idx = np.sort(rng.choice(train_idx, size=args.sample, replace=False))
    print(
        f"[bench] fold={fold_id} train={len(train_idx)} val={len(val_idx)}",
        flush=True,
    )

    X_train, y_train = X_full[train_idx], y[train_idx]
    X_val, y_val = X_full[val_idx], y[val_idx]

    # --- Black Box (CatBoost) ---
    t = _t()
    bb = CatBoostQuartetModel(
        iterations=_CATBOOST_ITERATIONS,
        learning_rate=_CATBOOST_LR,
        depth=_CATBOOST_DEPTH,
        seed=_SEED,
        thread_count=2,
    )
    bb.fit(X_train, y_train, cat_feature_indices=full_cat_idx or None)
    _report("catboost fit", t)
    t = _t()
    cb_pred_val = bb.predict(X_val)
    cb_pred_train = bb.predict(X_train)
    _report("catboost predict (val+train)", t)

    # --- Naive linear ---
    t = _t()
    nl = NaiveLinearQuartetModel()
    nl.fit(X_naive[train_idx], y_train, cat_feature_indices=naive_cat_idx or None)
    nl.predict(X_naive[val_idx])
    _report("naive_linear fit+predict", t)

    # --- Grey Box (DecisionTree on Black-OOF-style target) ---
    t = _t()
    grey = GreyTreeQuartetModel(max_depth=_GREY_MAX_DEPTH, seed=_SEED)
    grey.fit(X_train, cb_pred_train, cat_feature_indices=full_cat_idx or None)
    grey.predict(X_val)
    _report("grey_tree fit+predict", t)

    # --- White Box (EBM) — last, expected hog ---
    t = _t()
    wb = EbmQuartetModel(
        max_bins=_EBM_MAX_BINS,
        interactions=_EBM_INTERACTIONS,
        n_jobs=1,
    )
    wb.fit(X_train, y_train, cat_feature_indices=full_cat_idx or None)
    _report(
        f"ebm fit (max_bins={_EBM_MAX_BINS}, interactions={_EBM_INTERACTIONS}, n_jobs=1)",
        t,
    )
    t = _t()
    wb.predict(X_val)
    _report("ebm predict", t)

    # Cheap sanity metrics so the run isn't purely about timing.
    for name, pred in (("catboost", cb_pred_val),):
        mae = float(np.mean(np.abs(y_val - pred)))
        print(f"[bench] sanity {name} fold-{fold_id} MAE={mae:.2f}", flush=True)
    print(f"[bench] TOTAL {_t() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
