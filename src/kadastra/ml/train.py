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


def train_catboost(
    X: np.ndarray,
    y: np.ndarray,
    params: CatBoostParams,
    *,
    cat_features: list[int] | None = None,
) -> CatBoostRegressor:
    model = CatBoostRegressor(
        iterations=params.iterations,
        learning_rate=params.learning_rate,
        depth=params.depth,
        random_seed=params.seed,
        verbose=False,
        allow_writing_files=False,
        cat_features=cat_features,
    )
    model.fit(X, y, cat_features=cat_features)
    return model


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    h3_indices: list[str],
    *,
    params: CatBoostParams,
    n_splits: int,
    parent_resolution: int,
    cat_features: list[int] | None = None,
) -> dict[str, list[float] | list[int] | float]:
    """Spatial-CV training. In addition to per-fold + aggregate metrics,
    collects out-of-fold predictions: for each input row, the prediction
    came from a model that did **not** see that row in training. This
    is the honest spatial-CV view of the model's behavior over the
    full dataset and feeds the per-object inspector / map.

    Returns a dict with:

    - ``fold_mae`` / ``fold_rmse`` / ``fold_mape`` — per-fold metrics
    - ``mean_mae`` / ``mean_rmse`` / ``mean_mape`` — aggregate metrics
    - ``oof_indices`` — list of original row indices (unsorted; same
      order as folds were produced)
    - ``oof_fold_ids`` — fold id (0..n_splits-1) for each OOF index
    - ``oof_y_pred`` — predicted y for each OOF index

    The three OOF lists are parallel and have length ``len(y)`` when
    every row is in exactly one validation fold (the standard k-fold
    case for ``spatial_kfold_split``).
    """
    folds = spatial_kfold_split(h3_indices, n_splits=n_splits, parent_resolution=parent_resolution, seed=params.seed)

    fold_mae: list[float] = []
    fold_rmse: list[float] = []
    fold_mape: list[float] = []
    oof_indices: list[int] = []
    oof_fold_ids: list[int] = []
    oof_y_pred: list[float] = []

    for fold_id, (train_idx, val_idx) in enumerate(folds):
        model = train_catboost(X[train_idx], y[train_idx], params, cat_features=cat_features)
        preds = np.asarray(model.predict(X[val_idx]), dtype=np.float64)
        m = regression_metrics(y[val_idx], preds)
        fold_mae.append(m["mae"])
        fold_rmse.append(m["rmse"])
        fold_mape.append(m["mape"])
        oof_indices.extend(int(i) for i in val_idx)
        oof_fold_ids.extend(fold_id for _ in val_idx)
        oof_y_pred.extend(float(p) for p in preds)

    return {
        "fold_mae": fold_mae,
        "fold_rmse": fold_rmse,
        "fold_mape": fold_mape,
        "mean_mae": float(np.mean(fold_mae)),
        "mean_rmse": float(np.mean(fold_rmse)),
        "mean_mape": float(np.nanmean(fold_mape)),
        "oof_indices": oof_indices,
        "oof_fold_ids": oof_fold_ids,
        "oof_y_pred": oof_y_pred,
    }
