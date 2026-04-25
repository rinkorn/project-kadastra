from dataclasses import dataclass

import numpy as np
from catboost import CatBoostRegressor

from kadastra.ml.metrics import regression_metrics
from kadastra.ml.spatial_kfold import spatial_kfold_split


@dataclass(frozen=True)
class CatBoostParams:
    iterations: int
    learning_rate: float
    depth: int
    seed: int


def train_catboost(X: np.ndarray, y: np.ndarray, params: CatBoostParams) -> CatBoostRegressor:
    model = CatBoostRegressor(
        iterations=params.iterations,
        learning_rate=params.learning_rate,
        depth=params.depth,
        random_seed=params.seed,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(X, y)
    return model


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    h3_indices: list[str],
    *,
    params: CatBoostParams,
    n_splits: int,
    parent_resolution: int,
) -> dict[str, list[float] | float]:
    folds = spatial_kfold_split(
        h3_indices, n_splits=n_splits, parent_resolution=parent_resolution, seed=params.seed
    )

    fold_mae: list[float] = []
    fold_rmse: list[float] = []
    fold_mape: list[float] = []

    for train_idx, val_idx in folds:
        model = train_catboost(X[train_idx], y[train_idx], params)
        preds = np.asarray(model.predict(X[val_idx]), dtype=np.float64)
        m = regression_metrics(y[val_idx], preds)
        fold_mae.append(m["mae"])
        fold_rmse.append(m["rmse"])
        fold_mape.append(m["mape"])

    return {
        "fold_mae": fold_mae,
        "fold_rmse": fold_rmse,
        "fold_mape": fold_mape,
        "mean_mae": float(np.mean(fold_mae)),
        "mean_rmse": float(np.mean(fold_rmse)),
        "mean_mape": float(np.nanmean(fold_mape)),
    }
