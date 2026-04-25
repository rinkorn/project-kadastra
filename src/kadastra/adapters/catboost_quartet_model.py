"""Black Box adapter for the ADR-0016 quartet — wraps CatBoost.

Thin glue around the existing ``ml/train.train_catboost`` helper:
CatBoost natively eats the (numeric_first + categorical_last as
strings) matrix that the rest of the pipeline produces — no
preprocessing needed, ``cat_features`` indices are passed straight
through.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from catboost import CatBoostRegressor


class CatBoostQuartetModel:
    def __init__(
        self,
        *,
        iterations: int = 500,
        learning_rate: float = 0.05,
        depth: int = 6,
        seed: int = 42,
    ) -> None:
        self._iterations = iterations
        self._learning_rate = learning_rate
        self._depth = depth
        self._seed = seed
        self._model: CatBoostRegressor | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        cat_feature_indices: list[int] | None = None,
    ) -> None:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def serialize(self) -> bytes:
        raise NotImplementedError

    @classmethod
    def deserialize(cls, blob: bytes) -> CatBoostQuartetModel:
        raise NotImplementedError
