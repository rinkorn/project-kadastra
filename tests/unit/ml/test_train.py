import h3
import numpy as np
import pytest
from catboost import CatBoostRegressor

from kadastra.ml.train import CatBoostParams, cross_validate, train_catboost

KAZAN_LAT, KAZAN_LON = 55.7887, 49.1221


def _toy_dataset(n: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 3))
    # y = linear combo + noise → CatBoost should fit reasonably
    y = (2 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2] + rng.normal(scale=0.1, size=n)).astype(np.float64)
    cells = [h3.latlng_to_cell(KAZAN_LAT + 0.01 * i, KAZAN_LON, 8) for i in range(n)]
    return X, y, cells


_FAST_PARAMS = CatBoostParams(iterations=20, learning_rate=0.3, depth=4, seed=42)


def test_train_catboost_returns_fitted_regressor() -> None:
    X, y, _ = _toy_dataset(50)

    model = train_catboost(X, y, _FAST_PARAMS)

    assert isinstance(model, CatBoostRegressor)
    preds = model.predict(X)
    assert preds.shape == (50,)


def test_train_catboost_is_deterministic_given_seed() -> None:
    X, y, _ = _toy_dataset(50)

    p1 = train_catboost(X, y, _FAST_PARAMS).predict(X)
    p2 = train_catboost(X, y, _FAST_PARAMS).predict(X)

    np.testing.assert_array_equal(p1, p2)


def test_cross_validate_returns_per_fold_and_aggregate_metrics() -> None:
    X, y, cells = _toy_dataset(60)

    result = cross_validate(X, y, cells, params=_FAST_PARAMS, n_splits=3, parent_resolution=6)

    assert "fold_mae" in result and isinstance(result["fold_mae"], list)
    assert len(result["fold_mae"]) == 3  # type: ignore[arg-type]
    assert "mean_mae" in result and isinstance(result["mean_mae"], float)
    assert "mean_rmse" in result and isinstance(result["mean_rmse"], float)
    assert "mean_mape" in result and isinstance(result["mean_mape"], float)


def test_cross_validate_is_deterministic_given_params() -> None:
    X, y, cells = _toy_dataset(60)

    r1 = cross_validate(X, y, cells, params=_FAST_PARAMS, n_splits=3, parent_resolution=6)
    r2 = cross_validate(X, y, cells, params=_FAST_PARAMS, n_splits=3, parent_resolution=6)

    assert r1 == r2


def test_cross_validate_propagates_value_error_when_too_few_parents() -> None:
    # All cells at the same Kazan point → 1 parent → cannot split into 3
    X, y, _ = _toy_dataset(30)
    same_cell = h3.latlng_to_cell(KAZAN_LAT, KAZAN_LON, 8)
    cells = [same_cell] * 30

    with pytest.raises(ValueError, match="unique parents"):
        cross_validate(X, y, cells, params=_FAST_PARAMS, n_splits=3, parent_resolution=6)
