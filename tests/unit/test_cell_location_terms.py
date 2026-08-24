"""Tests for EBM term decomposition into location score (ADR-0029, product 2)."""

from __future__ import annotations

import numpy as np
import pytest

from kadastra.adapters.ebm_quartet_model import EbmQuartetModel
from kadastra.ml.cell_location_terms import (
    is_object_feature,
    sum_location_terms,
    top_location_terms,
)


def test_is_object_feature_classification() -> None:
    # Object attributes and their derived / relative columns.
    assert is_object_feature("area_m2")
    assert is_object_feature("levels")
    assert is_object_feature("materials")
    assert is_object_feature("vri")
    assert is_object_feature("era_category")
    assert is_object_feature("age_years")
    assert is_object_feature("polygon_area_m2")
    assert is_object_feature("area_m2__rel_p7_diff_med")
    assert is_object_feature("levels__rel_p8_z_iqr")
    # Locational.
    assert not is_object_feature("dist_to_cbd_m")
    assert not is_object_feature("dist_metro_m__rel_p7_diff_med")
    assert not is_object_feature("count_p7")
    assert not is_object_feature("nearest_road_class")
    assert not is_object_feature("oktmo_population")
    assert not is_object_feature("is_heritage_object")


def test_sum_location_terms_excludes_object_and_mixed_terms() -> None:
    term_values = np.array(
        [
            [1.0, 10.0, 100.0, 1000.0],
            [2.0, 20.0, 200.0, 2000.0],
        ]
    )
    term_features = [
        ("dist_to_cbd_m",),  # locational
        ("area_m2",),  # object
        ("levels", "dist_to_cbd_m"),  # mixed → excluded
        ("count_p7",),  # locational
    ]
    scores = sum_location_terms(term_values, term_features, intercept=5.0)
    assert scores[0] == pytest.approx(5.0 + 1.0 + 1000.0)
    assert scores[1] == pytest.approx(5.0 + 2.0 + 2000.0)


def test_top_location_terms_ranks_filters_and_joins_pair_names() -> None:
    term_values = np.array(
        [
            [1.0, 10.0, 100.0, 1000.0, 3.0],
            [2.0, 20.0, 200.0, 2000.0, 4.0],
        ]
    )
    term_features = [
        ("dist_to_cbd_m",),  # locational
        ("area_m2",),  # object → excluded
        ("levels", "dist_to_cbd_m"),  # mixed → excluded
        ("count_p7",),  # locational
        ("dist_metro_m", "walk_dist_to_school_m"),  # locational pair
    ]
    top = top_location_terms(term_values, term_features, top_n=2)
    assert top[0] == [
        {"feature": "count_p7", "contribution": 1000.0},
        {"feature": "dist_metro_m × walk_dist_to_school_m", "contribution": 3.0},
    ]
    assert top[1][0] == {"feature": "count_p7", "contribution": 2000.0}
    # top_n above the eligible count → all eligible terms.
    top_all = top_location_terms(term_values, term_features, top_n=10)
    assert len(top_all[0]) == 3


def test_top_location_terms_without_locational_terms_returns_empty() -> None:
    top = top_location_terms(np.ones((2, 1)), [("area_m2",)], top_n=5)
    assert top == [[], []]


def _fit_small_ebm() -> tuple[EbmQuartetModel, np.ndarray, list[str]]:
    rng = np.random.default_rng(42)
    n = 200
    dist = rng.uniform(0, 30_000, n)
    area = rng.uniform(30, 120, n)
    mat = rng.choice(["brick", "panel"], n)
    X = np.empty((n, 3), dtype=object)
    X[:, 0] = dist
    X[:, 1] = area
    X[:, 2] = mat
    y = 50_000 - dist + 100 * area + np.where(mat == "brick", 5_000, 0.0)
    model = EbmQuartetModel(interactions=2)
    model.fit(X, y, cat_feature_indices=[2])
    return model, X, ["dist_to_cbd_m", "area_m2", "materials"]


def test_eval_terms_plus_intercept_equals_predict() -> None:
    model, X, _ = _fit_small_ebm()
    terms = model.eval_terms(X)
    assert terms.shape[0] == X.shape[0]
    assert terms.shape[1] == len(model.term_feature_indices())
    reconstructed = model.intercept() + terms.sum(axis=1)
    np.testing.assert_allclose(reconstructed, model.predict(X), rtol=1e-9)


def test_location_score_from_real_ebm() -> None:
    model, X, feature_names = _fit_small_ebm()
    terms = model.eval_terms(X)
    term_features = [tuple(feature_names[i] for i in idxs) for idxs in model.term_feature_indices()]
    scores = sum_location_terms(terms, term_features, model.intercept())
    # dist_to_cbd_m is the only locational feature here; the score must
    # correlate perfectly with -dist (monotone decreasing shape).
    order = np.argsort(X[:, 0].astype(float))
    assert np.all(np.diff(scores[order]) <= 1e-6)


def test_eval_terms_before_fit_raises() -> None:
    model = EbmQuartetModel()
    with pytest.raises(RuntimeError):
        model.eval_terms(np.zeros((1, 1)))
    with pytest.raises(RuntimeError):
        model.intercept()
    with pytest.raises(RuntimeError):
        model.term_feature_indices()
