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
    y = np.array([float(numeric[i, 0]) + cat_to_y[str(cats[i, 0])] for i in range(n)])
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


def test_ebm_explain_returns_intercept_and_terms() -> None:
    """explain must decompose a prediction into intercept + per-term
    contributions that sum back to predict(X), with real feature names
    mapped from the matrix column order."""
    rng = np.random.default_rng(5)
    X = rng.normal(size=(100, 2))
    y = X[:, 0] * 1.5 + X[:, 1] * 0.5
    model = EbmQuartetModel(max_bins=32, interactions=0)
    model.fit(X, y, cat_feature_indices=None)
    feature_names = ["area_m2", "dist_metro_m"]
    out = model.explain(X[:1], feature_names)
    assert set(out) == {"intercept", "terms"}
    assert {t["feature"] for t in out["terms"]} == {"area_m2", "dist_metro_m"}
    total = out["intercept"] + sum(t["contribution"] for t in out["terms"])
    np.testing.assert_allclose(total, model.predict(X[:1])[0], rtol=1e-4)


def test_ebm_interactions_exclude_bans_pairs_but_keeps_main() -> None:
    """interactions_exclude must strip every PAIR containing the banned
    feature while keeping its main effect term. High-cardinality
    categoricals (vri, kadnum_quarter) explode the pair tensor and
    overfit; their mains are informative and must stay."""
    rng = np.random.default_rng(6)
    n = 300
    numeric = rng.normal(size=(n, 2))
    cats = np.array([[f"c{i % 20}"] for i in range(n)], dtype=object)
    X = np.hstack([numeric.astype(object), cats])
    y = numeric[:, 0] + numeric[:, 1] * numeric[:, 0] + rng.normal(0, 0.01, n)
    names = ["area_m2", "dist_metro_m", "vri"]

    model = EbmQuartetModel(max_bins=32, interactions=5, interactions_exclude=("vri",))
    model.fit(X, y, cat_feature_indices=[2], feature_names=names)

    terms = model.term_feature_indices()
    pair_terms = [t for t in terms if len(t) == 2]
    # No pair touches the banned feature (index 2)…
    assert all(2 not in t for t in pair_terms)
    # …but its main effect term survives.
    assert (2,) in terms


def test_ebm_deserialize_accepts_legacy_3_tuple() -> None:
    """Pre-interactions_exclude artifacts pickled a 3-tuple
    (max_bins, interactions, model) — the inspector loads such runs,
    so deserialize must stay backward compatible."""
    import pickle

    rng = np.random.default_rng(7)
    X = rng.normal(size=(80, 2))
    y = X[:, 0] + X[:, 1]
    model = EbmQuartetModel(max_bins=32, interactions=0)
    model.fit(X, y, cat_feature_indices=None)
    # Rebuild the legacy layout from the current 4-tuple payload.
    max_bins, interactions, _exclude, inner = pickle.loads(model.serialize())
    legacy_blob = pickle.dumps((max_bins, interactions, inner))
    restored = EbmQuartetModel.deserialize(legacy_blob)
    np.testing.assert_allclose(model.predict(X), restored.predict(X))
