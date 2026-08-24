"""Probe: EBM fit wall-time vs (rows, interactions) on landplot fold-0 train.

Isolates the White Box (the quartet's hog) so we can price the two
candidate levers — interactions=0 and smaller train — without rerunning
the full quartet bench. Single-threaded by default.

Usage:
    nice -n 15 uv run python scripts/bench_ebm_scaling.py --rows 20000 --interactions 0
    nice -n 15 uv run python scripts/bench_ebm_scaling.py --rows 60000 --interactions 10
"""

from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")

import h3
import numpy as np
import polars as pl

from kadastra.adapters.ebm_quartet_model import EbmQuartetModel
from kadastra.ml.object_feature_columns import select_object_feature_columns
from kadastra.ml.object_feature_matrix import build_object_feature_matrix
from kadastra.ml.spatial_kfold import spatial_kfold_split

_PARQUET = "data/gold/valuation_objects/region=RU-KAZAN-AGG/asset_class=landplot/data.parquet"
_TARGET_COLUMN = "synthetic_target_rub_per_m2"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--interactions", type=int, default=10)
    parser.add_argument("--max-bins", type=int, default=256)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--exclude-from-interactions",
        nargs="*",
        default=[],
        metavar="COL",
        help="Ban all interaction pairs touching these columns "
        "(mains are kept). Uses ExplainableBoostingRegressor directly "
        "with an explicit exclude list.",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    df = pl.read_parquet(_PARQUET).drop_nulls(subset=[_TARGET_COLUMN])
    y = df[_TARGET_COLUMN].to_numpy().astype(np.float64)
    num, cat = select_object_feature_columns(df)
    cat_idx = list(range(len(num), len(num) + len(cat)))
    X = build_object_feature_matrix(df, numeric_cols=num, categorical_cols=cat)
    h3_indices = [
        h3.latlng_to_cell(float(lat), float(lon), 10)
        for lat, lon in zip(df["lat"].to_list(), df["lon"].to_list(), strict=True)
    ]
    folds = spatial_kfold_split(h3_indices, n_splits=5, parent_resolution=7, seed=42)
    train_idx = np.array(folds[0][0], dtype=np.int64)
    if args.rows < len(train_idx):
        rng = np.random.default_rng(42)
        train_idx = np.sort(rng.choice(train_idx, size=args.rows, replace=False))
    print(f"[probe] setup {time.perf_counter() - t0:.1f}s, train rows={len(train_idx)}", flush=True)

    t = time.perf_counter()
    if args.exclude_from_interactions:
        from interpret.glassbox import ExplainableBoostingRegressor

        feature_names = num + cat
        cat_set = set(cat_idx)
        feature_types = ["nominal" if i in cat_set else "continuous" for i in range(X.shape[1])]
        banned = {feature_names.index(c) for c in args.exclude_from_interactions}
        exclude = [tuple(sorted((i, j))) for i in banned for j in range(X.shape[1]) if j != i]
        model = ExplainableBoostingRegressor(
            feature_types=feature_types,
            max_bins=args.max_bins,
            interactions=args.interactions,
            exclude=exclude,
            n_jobs=args.n_jobs,
        )
        model.fit(X[train_idx], y[train_idx])
        dt = time.perf_counter() - t
        print(
            f"[probe] EBM rows={len(train_idx)} interactions={args.interactions} "
            f"max_bins={args.max_bins} n_jobs={args.n_jobs} "
            f"exclude_from_interactions={args.exclude_from_interactions}: {dt:.1f}s",
            flush=True,
        )
        pairs = [idxs for idxs in model.term_features_ if len(idxs) == 2]
        for a, b in pairs:
            print(f"[probe] interaction pair: {feature_names[a]} x {feature_names[b]}", flush=True)
        return

    wb = EbmQuartetModel(max_bins=args.max_bins, interactions=args.interactions, n_jobs=args.n_jobs)
    wb.fit(X[train_idx], y[train_idx], cat_feature_indices=cat_idx or None)
    dt = time.perf_counter() - t
    print(
        f"[probe] EBM rows={len(train_idx)} interactions={args.interactions} "
        f"max_bins={args.max_bins} n_jobs={args.n_jobs}: {dt:.1f}s",
        flush=True,
    )
    feature_names = num + cat
    pairs = [idxs for idxs in wb.term_feature_indices() if len(idxs) == 2]
    for a, b in pairs:
        print(f"[probe] interaction pair: {feature_names[a]} x {feature_names[b]}", flush=True)


if __name__ == "__main__":
    main()
