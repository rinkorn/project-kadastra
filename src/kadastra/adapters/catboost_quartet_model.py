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
        thread_count: int | None = None,
    ) -> None:
        self._iterations = iterations
        self._learning_rate = learning_rate
        self._depth = depth
        self._seed = seed
        # When TrainQuartet runs folds in parallel, callers pass
        # thread_count=1 here to keep total CPU usage bounded
        # (otherwise N folds × CatBoost's default all-cores oversubscribes).
        self._thread_count = thread_count
        self._model: CatBoostRegressor | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        cat_feature_indices: list[int] | None = None,
    ) -> None:
        # Default CatBoost thread_count is -1 ("all cores"). When the
        # caller pins us to 1 (parallel-folds outer loop), pass through.
        thread_count = self._thread_count if self._thread_count is not None else -1
        model = CatBoostRegressor(
            iterations=self._iterations,
            learning_rate=self._learning_rate,
            depth=self._depth,
            random_seed=self._seed,
            verbose=False,
            allow_writing_files=False,
            cat_features=cat_feature_indices or None,
            thread_count=thread_count,
        )
        model.fit(X, y, cat_features=cat_feature_indices or None)
        self._model = model

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("CatBoostQuartetModel.predict before fit")
        preds = self._model.predict(X)
        return np.asarray(preds, dtype=np.float64)

    def unwrap(self) -> CatBoostRegressor:
        """Expose the underlying CatBoostRegressor.

        TrainQuartet passes the final-fit Black Box through
        ModelRegistryPort.log_run as the run's primary ``model``;
        the registry adapters expect a CatBoostRegressor by type.
        """
        if self._model is None:
            raise RuntimeError("CatBoostQuartetModel.unwrap before fit")
        return self._model

    def serialize(self) -> bytes:
        if self._model is None:
            raise RuntimeError("CatBoostQuartetModel.serialize before fit")
        with tempfile.NamedTemporaryFile(suffix=".cbm", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            self._model.save_model(str(tmp_path), format="cbm")
            return tmp_path.read_bytes()
        finally:
            tmp_path.unlink(missing_ok=True)

    @classmethod
    def deserialize(cls, blob: bytes) -> CatBoostQuartetModel:
        with tempfile.NamedTemporaryFile(suffix=".cbm", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            tmp_path.write_bytes(blob)
        try:
            model = CatBoostRegressor()
            model.load_model(str(tmp_path), format="cbm")
        finally:
            tmp_path.unlink(missing_ok=True)
        instance = cls()
        instance._model = model
        return instance
