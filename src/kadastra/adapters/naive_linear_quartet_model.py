"""Naive linear lower-bound model for the ADR-0016 quartet.

LinearRegression on top of:
- median-impute for the numeric columns (LinearRegression can't take
  NaN, but the per-object matrix does carry NaN for missing numerics);
- OneHotEncoder for the categorical columns with
  ``handle_unknown='ignore'`` so unseen-in-fit categories don't break
  predict (rare values land at the spatial-CV val boundary regularly).

This is *the* nothing-fancy baseline: no relative features, no
buffer-share, no derived parent-h3 aggregates — just the raw matrix
the rest of the quartet sees.
"""

from __future__ import annotations

import pickle

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


class NaiveLinearQuartetModel:
    def __init__(self) -> None:
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
            transformers.append(
                ("num", SimpleImputer(strategy="median"), num_idx)
            )
        if cat_idx:
            transformers.append(
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    cat_idx,
                )
            )

        preprocessor = ColumnTransformer(transformers)
        pipeline = Pipeline(
            steps=[("pre", preprocessor), ("lr", LinearRegression())]
        )
        pipeline.fit(X, y)
        self._pipeline = pipeline

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._pipeline is None:
            raise RuntimeError("NaiveLinearQuartetModel.predict before fit")
        preds = self._pipeline.predict(X)
        return np.asarray(preds, dtype=np.float64)

    def serialize(self) -> bytes:
        if self._pipeline is None:
            raise RuntimeError("NaiveLinearQuartetModel.serialize before fit")
        return pickle.dumps(self._pipeline)

    @classmethod
    def deserialize(cls, blob: bytes) -> NaiveLinearQuartetModel:
        instance = cls()
        instance._pipeline = pickle.loads(blob)
        return instance
