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

import numpy as np


class GreyTreeQuartetModel:
    def __init__(self, *, max_depth: int = 10) -> None:
        self._max_depth = max_depth

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
    def deserialize(cls, blob: bytes) -> GreyTreeQuartetModel:
        raise NotImplementedError
