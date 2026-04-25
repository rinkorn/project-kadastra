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

import numpy as np


class NaiveLinearQuartetModel:
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
    def deserialize(cls, blob: bytes) -> NaiveLinearQuartetModel:
        raise NotImplementedError
