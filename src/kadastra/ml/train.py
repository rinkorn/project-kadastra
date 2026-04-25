from dataclasses import dataclass

import numpy as np
from catboost import CatBoostRegressor


@dataclass(frozen=True)
class CatBoostParams:
    iterations: int
    learning_rate: float
    depth: int
    seed: int


def train_catboost(X: np.ndarray, y: np.ndarray, params: CatBoostParams) -> CatBoostRegressor:
    raise NotImplementedError


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    h3_indices: list[str],
    *,
    params: CatBoostParams,
    n_splits: int,
    parent_resolution: int,
) -> dict[str, list[float] | float]:
    raise NotImplementedError
