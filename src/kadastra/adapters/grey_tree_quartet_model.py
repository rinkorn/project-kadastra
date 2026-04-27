"""Grey Box approximator for the ADR-0016 quartet.

DecisionTreeRegressor (sklearn) bounded to ``max_depth`` so the tree
stays mid-complexity — its job is to *approximate* the Black Box, not
to compete with it on raw fit. Trained on Black Box's OOF predictions
(plumbed in by TrainQuartet), which lets us measure how well a simple
tree can copy the Black Box.

Adapter is target-agnostic: caller passes whatever ``y`` they want as
the training target.
"""

from __future__ import annotations

import pickle

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeRegressor


class GreyTreeQuartetModel:
    def __init__(self, *, max_depth: int = 10, seed: int = 42) -> None:
        self._max_depth = max_depth
        self._seed = seed
        self._pipeline: Pipeline | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        cat_feature_indices: list[int] | None = None,
    ) -> None:
        cat_idx = list(cat_feature_indices or [])
        n_cols = X.shape[1]
        num_idx = [i for i in range(n_cols) if i not in set(cat_idx)]

        transformers: list[tuple[str, object, list[int]]] = []
        if num_idx:
            # Pass-through; sklearn ≥ 1.3 trees handle NaN natively.
            transformers.append(("num", "passthrough", num_idx))
        if cat_idx:
            transformers.append(
                (
                    "cat",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                    cat_idx,
                )
            )

        preprocessor = ColumnTransformer(transformers)
        regressor = DecisionTreeRegressor(
            max_depth=self._max_depth,
            random_state=self._seed,
        )
        pipeline = Pipeline(steps=[("pre", preprocessor), ("dt", regressor)])
        pipeline.fit(X, y)
        self._pipeline = pipeline

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._pipeline is None:
            raise RuntimeError("GreyTreeQuartetModel.predict before fit")
        preds = self._pipeline.predict(X)
        return np.asarray(preds, dtype=np.float64)

    def serialize(self) -> bytes:
        if self._pipeline is None:
            raise RuntimeError("GreyTreeQuartetModel.serialize before fit")
        return pickle.dumps((self._max_depth, self._seed, self._pipeline))

    @classmethod
    def deserialize(cls, blob: bytes) -> GreyTreeQuartetModel:
        max_depth, seed, pipeline = pickle.loads(blob)
        instance = cls(max_depth=max_depth, seed=seed)
        instance._pipeline = pipeline
        return instance
