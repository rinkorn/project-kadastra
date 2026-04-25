"""Port for the Black/Grey/White/Naive quartet (ADR-0016).

Each model in the quartet is wrapped behind a single small Protocol:

- ``fit(X, y, *, cat_feature_indices)`` — train on the same matrix
  layout that ``build_object_feature_matrix`` produces (numeric
  columns first as floats with NaN for missing; categorical columns
  last as Python strings with a sentinel for missing).
- ``predict(X)`` — score a matrix of the same shape.
- ``serialize()`` — return a ``bytes`` artifact for the model
  registry. Adapters pick a sensible format per backend (cbm for
  CatBoost, pickle for sklearn / interpret-ml).

The port deliberately *receives* a matrix that already has its
categories as strings, leaving each adapter to encode them
appropriately (CatBoost uses ``cat_feature_indices`` directly, EBM
declares feature types, sklearn-based adapters wrap ``OneHotEncoder``
or ``OrdinalEncoder``).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class QuartetModelPort(Protocol):
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        cat_feature_indices: list[int] | None = None,
    ) -> None: ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...

    def serialize(self) -> bytes: ...
