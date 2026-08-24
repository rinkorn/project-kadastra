"""A/B: do EBM pairwise interactions actually help on landplot?

Arm A — interactions=0 (mains only). Arm B — interactions=N with the
high-cardinality categoricals (vri, kadnum_quarter) banned from pairs
via the new ``interactions_exclude`` adapter knob. Both arms train on
the same deterministic subsample of fold-0 train and are scored on the
full fold-0 validation: MAPE / WAPE / Spearman + fit wall-time.

Usage:
    nice -n 15 uv run python scripts/bench_ebm_interactions_value.py
    nice -n 15 uv run python scripts/bench_ebm_interactions_value.py --rows 50000 --interactions-b 5
"""

from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")

import h3
import numpy as np
import polars as pl

from kadastra.adapters.ebm_quartet_model import EbmQuartetModel
from kadastra.ml.object_feature_columns import select_object_feature_columns
from kadastra.ml.object_feature_matrix import build_object_feature_matrix
from kadastra.ml.quartet_metrics import spearman_corr, wape
from kadastra.ml.spatial_kfold import spatial_kfold_split

_PARQUET = "data/gold/valuation_objects/region=RU-KAZAN-AGG/asset_class=landplot/data.parquet"
_TARGET_COLUMN = "synthetic_target_rub_per_m2"
_EXCLUDE = ("vri", "kadnum_quarter")


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mask = y_true > 0
    mape = float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / y_true[mask]))
    return {
        "mape": mape,
        "wape": wape(y_true, y_pred),
        "spearman": spearman_corr(y_true, y_pred),
    }


def _run_arm(
    label: str,
    wb: EbmQuartetModel,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cat_idx: list[int],
    feature_names: list[str],
) -> None:
    t = time.perf_counter()
    wb.fit(X_train, y_train, cat_feature_indices=cat_idx or None, feature_names=feature_names)
    fit_s = time.perf_counter() - t
    pred = wb.predict(X_val)
    m = _metrics(y_val, pred)
    print(
        f"[ab] {label}: fit={fit_s:.1f}s mape={m['mape']:.4f} wape={m['wape']:.4f} spearman={m['spearman']:.4f}",
        flush=True,
    )
    pairs = [idxs for idxs in wb.term_feature_indices() if len(idxs) == 2]
    for a, b in pairs:
        print(f"[ab] {label} pair: {feature_names[a]} x {feature_names[b]}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=50000)
    parser.add_argument("--interactions-b", type=int, default=5)
    parser.add_argument("--max-bins", type=int, default=256)
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()

    t0 = time.perf_counter()
    df = pl.read_parquet(_PARQUET).drop_nulls(subset=[_TARGET_COLUMN])
    y = df[_TARGET_COLUMN].to_numpy().astype(np.float64)
    num, cat = select_object_feature_columns(df)
    feature_names = num + cat
    cat_idx = list(range(len(num), len(feature_names)))
    X = build_object_feature_matrix(df, numeric_cols=num, categorical_cols=cat)
    h3_indices = [
        h3.latlng_to_cell(float(lat), float(lon), 10)
        for lat, lon in zip(df["lat"].to_list(), df["lon"].to_list(), strict=True)
    ]
    folds = spatial_kfold_split(h3_indices, n_splits=5, parent_resolution=7, seed=42)
    train_idx = np.array(folds[0][0], dtype=np.int64)
    val_idx = np.array(folds[0][1], dtype=np.int64)
    if args.rows < len(train_idx):
        rng = np.random.default_rng(42)
        train_idx = np.sort(rng.choice(train_idx, size=args.rows, replace=False))
    print(
        f"[ab] setup {time.perf_counter() - t0:.1f}s, train={len(train_idx)} val={len(val_idx)}",
        flush=True,
    )

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    _run_arm(
        "A interactions=0",
        EbmQuartetModel(max_bins=args.max_bins, interactions=0, n_jobs=args.n_jobs),
        X_train,
        y_train,
        X_val,
        y_val,
        cat_idx,
        feature_names,
    )
    _run_arm(
        f"B interactions={args.interactions_b} exclude={list(_EXCLUDE)}",
        EbmQuartetModel(
            max_bins=args.max_bins,
            interactions=args.interactions_b,
            n_jobs=args.n_jobs,
            interactions_exclude=_EXCLUDE,
        ),
        X_train,
        y_train,
        X_val,
        y_val,
        cat_idx,
        feature_names,
    )
    print(f"[ab] TOTAL {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
