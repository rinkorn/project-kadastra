"""Train the Black/Grey/White/Naive quartet (ADR-0016) per AssetClass.

Each class trains in a fresh Python subprocess so joblib's loky
worker pool from one class can't bleed into the next — sequential
in-process runs caused ~10× slowdown by the third or fourth class
once feature matrices grew (POI / linear-distance / age fields).

Run name: quartet-object-{class}. Registry backend follows
MLFLOW_ENABLED. Each run carries the four models' OOF predictions,
serialized White/Grey/Naive models, and a quartet_metrics.json with
per-model MAE/RMSE/MAPE/Spearman + percentile asymmetry +
loss_on_simplification (in pp).

Usage:
    uv run python scripts/train_quartet.py
    uv run python scripts/train_quartet.py --asset-class commercial
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from kadastra.composition_root import Container
from kadastra.config import Settings
from kadastra.domain.asset_class import AssetClass


def _train_one(asset_class: AssetClass) -> None:
    settings = Settings()
    container = Container(settings)
    usecase = container.build_train_quartet()
    run_id = usecase.execute(settings.region_code, asset_class)
    print(f"  class={asset_class.value}: run_id={run_id}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--asset-class",
        choices=[c.value for c in AssetClass],
        help="Train this single class in the current process. "
             "Without this flag, spawns one fresh subprocess per class "
             "so loky worker pools don't leak between classes.",
    )
    args = parser.parse_args()

    if args.asset_class is not None:
        _train_one(AssetClass(args.asset_class))
        return

    settings = Settings()
    backend = "mlflow" if settings.mlflow_enabled else "local"
    classes = list(AssetClass)
    print(
        f"Training quartet (Black/White/Grey/Naive): "
        f"region={settings.region_code} "
        f"classes={[c.value for c in classes]} "
        f"catboost(iters={settings.catboost_iterations}, "
        f"lr={settings.catboost_learning_rate}, depth={settings.catboost_depth}) "
        f"ebm(max_bins={settings.ebm_max_bins}, "
        f"interactions={settings.ebm_interactions}) "
        f"grey(max_depth={settings.grey_tree_max_depth}) "
        f"n_splits={settings.train_n_splits} "
        f"parent_res={settings.train_parent_resolution} backend={backend} "
        f"parallel_folds={settings.quartet_parallel_folds} "
        f"skip_final_fits={settings.quartet_skip_final_simplifier_fits} "
        f"per_class_subprocess=True",
        flush=True,
    )

    for asset_class in classes:
        result = subprocess.run(
            [sys.executable, __file__, "--asset-class", asset_class.value],
            check=False,
        )
        if result.returncode != 0:
            sys.exit(result.returncode)


if __name__ == "__main__":
    main()
