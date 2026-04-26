"""Tests for NaiveLinearQuartetModel — the lower-bound model in the
ADR-0016 quartet (LinearRegression + OneHotEncoder + SimpleImputer).

The naive baseline is fed the same (numeric_first + categorical_last)
matrix produced by ``build_object_feature_matrix``, so the adapter
itself owns its preprocessing — median-impute the numerics
(LinearRegression can't take NaN), one-hot the categories with
``handle_unknown='ignore'`` (so a category seen only in val doesn't
break predict).

These tests are intentionally light on numerics: we don't check that
the model fits well, only that it fits *at all* on the matrix layout
the rest of the pipeline produces, and that the surface contract
(serialize → deserialize round-trip, unseen-category robustness) holds.
"""

from __future__ import annotations

import numpy as np

from kadastra.adapters.naive_linear_quartet_model import NaiveLinearQuartetModel


def test_naive_linear_fits_pure_numeric() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    y = X[:, 0] * 2.0 + X[:, 1] * 0.5 + rng.normal(size=100) * 0.1
    model = NaiveLinearQuartetModel()
    model.fit(X, y, cat_feature_indices=None)
    preds = model.predict(X)
    assert preds.shape == (100,)
    # Sanity: with such a clean signal R² should be > 0.95.
    ss_res = float(np.sum((y - preds) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    assert 1 - ss_res / ss_tot > 0.95


def test_naive_linear_handles_missing_numeric() -> None:
    """LinearRegression itself can't take NaN — the adapter wraps a
    SimpleImputer so NaN-bearing rows go through fit without raising."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 2))
    X[5, 0] = float("nan")
    X[20, 1] = float("nan")
    y = rng.normal(size=50)
    model = NaiveLinearQuartetModel()
    model.fit(X, y, cat_feature_indices=None)
    preds = model.predict(X)
    assert preds.shape == (50,)
    assert not np.isnan(preds).any()


def test_naive_linear_handles_categorical_via_one_hot() -> None:
    """X has 1 numeric col + 1 categorical col (last index). The
    adapter builds a OneHotEncoder over the cat column and trains
    a LinearRegression over [imputed_numeric, one_hot_cat]."""
    n = 60
    rng = np.random.default_rng(2)
    numeric = rng.normal(size=(n, 1))
    cats = np.array(
        [["red"], ["blue"], ["green"]] * (n // 3), dtype=object
    )
    X = np.hstack([numeric.astype(object), cats])
    # Encode a clean signal where each category shifts y by a fixed amount.
    cat_to_y = {"red": 0.0, "blue": 1.0, "green": 2.0}
    y = np.array(
        [
            float(numeric[i, 0]) + cat_to_y[str(cats[i, 0])]
            for i in range(n)
        ]
    )
    model = NaiveLinearQuartetModel()
    model.fit(X, y, cat_feature_indices=[1])
    preds = model.predict(X)
    assert preds.shape == (n,)
    # Should fit the structured signal extremely well.
    rmse = float(np.sqrt(np.mean((preds - y) ** 2)))
    assert rmse < 0.05


def test_naive_linear_predicts_on_unseen_category() -> None:
    """If predict() sees a category that fit() didn't, the adapter
    must still produce a number (one-hot with handle_unknown='ignore')."""
    n = 30
    rng = np.random.default_rng(3)
    numeric = rng.normal(size=(n, 1))
    cats = np.array([["a"], ["b"]] * (n // 2), dtype=object)
    X_train = np.hstack([numeric.astype(object), cats])
    y = rng.normal(size=n)
    model = NaiveLinearQuartetModel()
    model.fit(X_train, y, cat_feature_indices=[1])
    # Predict on a row whose cat is "c" — never seen in fit.
    X_unseen = np.array([[0.5, "c"]], dtype=object)
    preds = model.predict(X_unseen)
    assert preds.shape == (1,)
    assert not np.isnan(preds[0])


def test_naive_linear_serialize_round_trip() -> None:
    rng = np.random.default_rng(4)
    X = rng.normal(size=(40, 2))
    y = X[:, 0] + X[:, 1]
    model = NaiveLinearQuartetModel()
    model.fit(X, y, cat_feature_indices=None)
    blob = model.serialize()
    assert isinstance(blob, bytes)
    assert len(blob) > 0
    restored = NaiveLinearQuartetModel.deserialize(blob)
    np.testing.assert_allclose(model.predict(X), restored.predict(X))
