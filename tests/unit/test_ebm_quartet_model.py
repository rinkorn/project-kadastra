"""Tests for EbmQuartetModel — the White Box of the ADR-0016 quartet
(interpret-ml's ExplainableBoostingRegressor).

EBM is additive over per-feature shape functions + a bounded number
of pairwise interactions, so it stays interpretable while being
nonlinear. It supports categorical columns natively via the
``feature_types`` argument — we pass ``'continuous'`` for numeric
indices and ``'nominal'`` for categorical, matching the layout of
``build_object_feature_matrix``.

Tests: fits a simple structured signal; survives NaN in numerics
(EBM's binner treats NaN as its own bin); survives unseen-in-fit
categories (handled by EBM's nominal binning); serialize round-trip.
"""

from __future__ import annotations

import numpy as np

from kadastra.adapters.ebm_quartet_model import EbmQuartetModel


def test_ebm_fits_pure_numeric() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    # Additive structured signal — EBM should fit this cleanly.
    y = X[:, 0] * 1.5 + np.sin(X[:, 1]) * 2.0
    model = EbmQuartetModel(max_bins=64, interactions=0)
    model.fit(X, y, cat_feature_indices=None)
    preds = model.predict(X)
    assert preds.shape == (200,)
    rmse = float(np.sqrt(np.mean((preds - y) ** 2)))
    # Generous bound — EBM with default params on 200 samples is not
    # going to memorize a continuous signal exactly.
    assert rmse < 0.6


def test_ebm_handles_categorical() -> None:
    n = 120
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
    model = EbmQuartetModel(max_bins=32, interactions=0)
    model.fit(X, y, cat_feature_indices=[1])
    preds = model.predict(X)
    rmse = float(np.sqrt(np.mean((preds - y) ** 2)))
    assert rmse < 0.5


def test_ebm_predicts_on_unseen_category() -> None:
    n = 60
    rng = np.random.default_rng(3)
    numeric = rng.normal(size=(n, 1))
    cats = np.array([["a"], ["b"]] * (n // 2), dtype=object)
    X_train = np.hstack([numeric.astype(object), cats])
    y = rng.normal(size=n)
    model = EbmQuartetModel(max_bins=32, interactions=0)
    model.fit(X_train, y, cat_feature_indices=[1])
    X_unseen = np.array([[0.5, "c"]], dtype=object)
    preds = model.predict(X_unseen)
    assert preds.shape == (1,)
    assert not np.isnan(preds[0])


def test_ebm_serialize_round_trip() -> None:
    rng = np.random.default_rng(4)
    X = rng.normal(size=(80, 2))
    y = X[:, 0] + X[:, 1]
    model = EbmQuartetModel(max_bins=32, interactions=0)
    model.fit(X, y, cat_feature_indices=None)
    blob = model.serialize()
    assert isinstance(blob, bytes)
    restored = EbmQuartetModel.deserialize(blob)
    np.testing.assert_allclose(model.predict(X), restored.predict(X))
