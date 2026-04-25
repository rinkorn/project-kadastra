"""Tests for GreyTreeQuartetModel — the Grey Box approximator in the
ADR-0016 quartet.

It's a sklearn DecisionTreeRegressor (max_depth bounded so the tree
stays interpretable-ish) wrapped behind the same fit/predict/serialize
contract. Unlike the rest of the quartet, the *intended* training
target is Black Box's OOF predictions, not y_true — but the adapter
itself is target-agnostic, so these tests exercise it like a regular
regressor. The use case (TrainQuartet) plumbs the y_pred_oof through.

Categorical columns are ordinal-encoded with
``handle_unknown='use_encoded_value', unknown_value=-1`` — DT splits
on -1 cleanly when an unseen category arrives at predict time.
Numerics flow through unchanged (sklearn 1.3+ trees handle NaN).
"""

from __future__ import annotations

import numpy as np

from kadastra.adapters.grey_tree_quartet_model import GreyTreeQuartetModel


def test_grey_tree_fits_pure_numeric() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 3))
    # Non-linear signal so a tree of depth 6 actually beats a linear fit.
    y = np.where(X[:, 0] > 0, X[:, 1] + 1.0, X[:, 2] - 1.0)
    model = GreyTreeQuartetModel(max_depth=6)
    model.fit(X, y, cat_feature_indices=None)
    preds = model.predict(X)
    assert preds.shape == (120,)
    rmse = float(np.sqrt(np.mean((preds - y) ** 2)))
    assert rmse < 0.5


def test_grey_tree_handles_missing_numeric() -> None:
    """sklearn ≥ 1.3 DecisionTreeRegressor supports NaN natively
    (sends rows down the side with the most training mass at each
    split). The adapter does not need its own imputer."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 2))
    X[3, 0] = float("nan")
    X[10, 1] = float("nan")
    y = rng.normal(size=60)
    model = GreyTreeQuartetModel(max_depth=4)
    model.fit(X, y, cat_feature_indices=None)
    preds = model.predict(X)
    assert preds.shape == (60,)
    assert not np.isnan(preds).any()


def test_grey_tree_handles_categorical_via_ordinal() -> None:
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
    model = GreyTreeQuartetModel(max_depth=6)
    model.fit(X, y, cat_feature_indices=[1])
    preds = model.predict(X)
    assert preds.shape == (n,)
    rmse = float(np.sqrt(np.mean((preds - y) ** 2)))
    # Tree of depth 6 can carve out the 3 categorical bins easily.
    assert rmse < 0.5


def test_grey_tree_predicts_on_unseen_category() -> None:
    n = 30
    rng = np.random.default_rng(3)
    numeric = rng.normal(size=(n, 1))
    cats = np.array([["a"], ["b"]] * (n // 2), dtype=object)
    X_train = np.hstack([numeric.astype(object), cats])
    y = rng.normal(size=n)
    model = GreyTreeQuartetModel(max_depth=4)
    model.fit(X_train, y, cat_feature_indices=[1])
    X_unseen = np.array([[0.5, "c"]], dtype=object)
    preds = model.predict(X_unseen)
    assert preds.shape == (1,)
    assert not np.isnan(preds[0])


def test_grey_tree_serialize_round_trip() -> None:
    rng = np.random.default_rng(4)
    X = rng.normal(size=(40, 2))
    y = X[:, 0] + X[:, 1]
    model = GreyTreeQuartetModel(max_depth=4)
    model.fit(X, y, cat_feature_indices=None)
    blob = model.serialize()
    assert isinstance(blob, bytes)
    restored = GreyTreeQuartetModel.deserialize(blob)
    np.testing.assert_allclose(model.predict(X), restored.predict(X))
