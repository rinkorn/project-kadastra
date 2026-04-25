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


class EbmQuartetModel:
    def __init__(self, *, max_bins: int = 256, interactions: int = 10) -> None:
        self._max_bins = max_bins
        self._interactions = interactions
        self._model = None

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
    def deserialize(cls, blob: bytes) -> EbmQuartetModel:
        raise NotImplementedError
