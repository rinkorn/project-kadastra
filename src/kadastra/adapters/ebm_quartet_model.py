"""White Box adapter for the ADR-0016 quartet — interpret-ml's
ExplainableBoostingRegressor.

EBM is an additive model over per-feature shape functions + a small
number of pairwise interactions. Categorical columns are declared via
``feature_types``: ``'continuous'`` for numerics, ``'nominal'`` for
categoricals. The binner handles NaN in numerics and unseen-in-fit
categories without explicit imputation/encoding.
"""

from __future__ import annotations

import pickle

import numpy as np
from interpret.glassbox import ExplainableBoostingRegressor


class EbmQuartetModel:
    def __init__(
        self,
        *,
        max_bins: int = 256,
        interactions: int = 10,
        n_jobs: int | None = None,
    ) -> None:
        self._max_bins = max_bins
        self._interactions = interactions
        # When TrainQuartet runs folds in parallel, callers pass
        # n_jobs=1 here so EBM's outer_bags stay sequential — a parallel
        # outer × parallel inner combination oversubscribes the machine.
        self._n_jobs = n_jobs
        self._model: ExplainableBoostingRegressor | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        cat_feature_indices: list[int] | None = None,
    ) -> None:
        cat_set = set(cat_feature_indices or [])
        feature_types = [
            "nominal" if i in cat_set else "continuous"
            for i in range(X.shape[1])
        ]
        kwargs: dict[str, object] = {
            "feature_types": feature_types,
            "max_bins": self._max_bins,
            "interactions": self._interactions,
        }
        if self._n_jobs is not None:
            kwargs["n_jobs"] = self._n_jobs
        model = ExplainableBoostingRegressor(**kwargs)
        model.fit(X, y)
        self._model = model

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("EbmQuartetModel.predict before fit")
        preds = self._model.predict(X)
        return np.asarray(preds, dtype=np.float64)

    def serialize(self) -> bytes:
        if self._model is None:
            raise RuntimeError("EbmQuartetModel.serialize before fit")
        return pickle.dumps(
            (self._max_bins, self._interactions, self._model)
        )

    @classmethod
    def deserialize(cls, blob: bytes) -> EbmQuartetModel:
        max_bins, interactions, model = pickle.loads(blob)
        instance = cls(max_bins=max_bins, interactions=interactions)
        instance._model = model
        return instance
