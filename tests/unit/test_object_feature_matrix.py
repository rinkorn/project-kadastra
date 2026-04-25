"""Unit tests for build_object_feature_matrix.

Pure helper used by both train and infer use cases. Numeric nulls flow
through as NaN (CatBoost handles missing natively); categorical nulls
become a sentinel string so the model treats "unknown" as its own
category instead of conflating it with a real value (e.g. year_built=0
would otherwise mean "ancient ruin", and materials=None would silently
collide with whatever category sorts to position 0).
"""

from __future__ import annotations

import math

import polars as pl

from kadastra.ml.object_feature_matrix import (
    MISSING_CATEGORY,
    build_object_feature_matrix,
)


def test_numeric_nulls_become_nan_not_zero() -> None:
    df = pl.DataFrame(
        {
            "year_built": [1990, None, 2010],
            "levels": [3, 5, None],
        },
        schema={"year_built": pl.Int64, "levels": pl.Int64},
    )

    X = build_object_feature_matrix(
        df, numeric_cols=["year_built", "levels"], categorical_cols=[]
    )

    # X is a (3, 2) numpy array; nulls -> NaN, not 0
    assert X.shape == (3, 2)
    assert math.isnan(X[1, 0])
    assert math.isnan(X[2, 1])
    assert X[0, 0] == 1990
    assert X[1, 1] == 5


def test_categorical_nulls_become_sentinel_string() -> None:
    df = pl.DataFrame(
        {"materials": ["Кирпичные", None, "Панельные"]},
        schema={"materials": pl.Utf8},
    )

    X = build_object_feature_matrix(
        df, numeric_cols=[], categorical_cols=["materials"]
    )

    assert X.shape == (3, 1)
    assert X[0, 0] == "Кирпичные"
    assert X[1, 0] == MISSING_CATEGORY
    assert X[2, 0] == "Панельные"


def test_combined_matrix_orders_numeric_then_categorical() -> None:
    df = pl.DataFrame(
        {
            "levels": [3, None],
            "year_built": [1990, 2010],
            "materials": ["Кирпичные", None],
        },
        schema={
            "levels": pl.Int64,
            "year_built": pl.Int64,
            "materials": pl.Utf8,
        },
    )

    X = build_object_feature_matrix(
        df,
        numeric_cols=["levels", "year_built"],
        categorical_cols=["materials"],
    )

    assert X.shape == (2, 3)
    # Row 0: numeric, numeric, categorical
    assert X[0, 0] == 3
    assert X[0, 1] == 1990
    assert X[0, 2] == "Кирпичные"
    # Row 1: NaN for missing level, real year, sentinel for missing materials
    assert math.isnan(X[1, 0])
    assert X[1, 1] == 2010
    assert X[1, 2] == MISSING_CATEGORY
