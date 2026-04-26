"""Tests for CatBoostQuartetModel — the Black Box of the ADR-0016
quartet (wraps the existing ``ml/train.train_catboost``).

The other three quartet adapters had to bring their own preprocessing.
CatBoost takes the (numeric_first + categorical_last) matrix
unchanged and uses ``cat_features`` indices natively, so this adapter
is little more than thin glue around the existing helper.
"""

from __future__ import annotations

import numpy as np

from kadastra.adapters.catboost_quartet_model import CatBoostQuartetModel


def test_catboost_fits_pure_numeric() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    y = X[:, 0] * 2.0 + X[:, 1] * 0.5 + rng.normal(size=200) * 0.05
    model = CatBoostQuartetModel(iterations=100, learning_rate=0.1, depth=4)
    model.fit(X, y, cat_feature_indices=None)
    preds = model.predict(X)
    assert preds.shape == (200,)
    rmse = float(np.sqrt(np.mean((preds - y) ** 2)))
    assert rmse < 0.5


def test_catboost_handles_categorical_natively() -> None:
    n = 90
    rng = np.random.default_rng(2)
    numeric = rng.normal(size=(n, 1))
    cats = np.array([["red"], ["blue"], ["green"]] * (n // 3), dtype=object)
    X = np.hstack([numeric.astype(object), cats])
    cat_to_y = {"red": 0.0, "blue": 1.0, "green": 2.0}
    y = np.array(
        [
            float(numeric[i, 0]) + cat_to_y[str(cats[i, 0])]
            for i in range(n)
        ]
    )
    model = CatBoostQuartetModel(iterations=100, learning_rate=0.1, depth=4)
    model.fit(X, y, cat_feature_indices=[1])
    preds = model.predict(X)
    rmse = float(np.sqrt(np.mean((preds - y) ** 2)))
    assert rmse < 0.5


def test_catboost_serialize_round_trip() -> None:
    rng = np.random.default_rng(4)
    X = rng.normal(size=(60, 2))
    y = X[:, 0] + X[:, 1]
    model = CatBoostQuartetModel(iterations=100, learning_rate=0.1, depth=4)
    model.fit(X, y, cat_feature_indices=None)
    blob = model.serialize()
    assert isinstance(blob, bytes)
    restored = CatBoostQuartetModel.deserialize(blob)
    np.testing.assert_allclose(model.predict(X), restored.predict(X))
