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
    prepared = df.with_columns(
        [pl.col(c).cast(pl.Float64) for c in numeric_cols]
        + [pl.col(c).fill_null(MISSING_CATEGORY).cast(pl.Utf8) for c in categorical_cols]
    )
    feature_cols = numeric_cols + categorical_cols
    if not feature_cols:
        return np.empty((prepared.height, 0))

    if not categorical_cols:
        # Pure numeric: float64 matrix with NaN for missing.
        return prepared.select(numeric_cols).to_numpy()

    if not numeric_cols:
        # Pure categorical: object matrix of strings.
        cells = [prepared[c].to_list() for c in categorical_cols]
        out = np.empty((prepared.height, len(categorical_cols)), dtype=object)
        for j, col_values in enumerate(cells):
            for i, v in enumerate(col_values):
                out[i, j] = v
        return out

    # Mixed: object matrix with floats (incl. NaN) and strings.
    out = np.empty((prepared.height, len(feature_cols)), dtype=object)
    for j, col in enumerate(numeric_cols):
        col_values = prepared[col].to_list()
        for i, v in enumerate(col_values):
            out[i, j] = float("nan") if v is None else float(v)
    for offset, col in enumerate(categorical_cols):
        j = len(numeric_cols) + offset
        col_values = prepared[col].to_list()
        for i, v in enumerate(col_values):
            out[i, j] = v
    return out
