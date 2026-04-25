"""Pure helper for materializing per-object feature matrices.

Both TrainObjectValuationModel and InferObjectValuation must hand the
same shape of matrix to CatBoost: numeric columns first (with NaN for
missing), categorical columns last as Python strings (with a sentinel
for missing). Centralizing the logic keeps train and infer in sync.
"""

from __future__ import annotations

import numpy as np
import polars as pl

MISSING_CATEGORY = "__missing__"


def build_object_feature_matrix(
    df: pl.DataFrame,
    *,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> np.ndarray:
    raise NotImplementedError
