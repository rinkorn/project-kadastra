from pathlib import Path

import numpy as np
import pytest

from kadastra.adapters.mlflow_model_loader import MLflowModelLoader
from kadastra.adapters.mlflow_model_registry import MLflowModelRegistry
from kadastra.ml.train import CatBoostParams, train_catboost


def _make_trained_model():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    y = X[:, 0] + 2 * X[:, 1] - X[:, 2]
    model = train_catboost(X, y, CatBoostParams(iterations=10, learning_rate=0.3, depth=3, seed=0))
    return model, X


def test_load_returns_model_with_same_predictions(tmp_path: Path) -> None:
    tracking_uri = f"file:{tmp_path / 'mlruns'}"
    registry = MLflowModelRegistry(tracking_uri=tracking_uri, experiment_name="test")
    model, X = _make_trained_model()
    run_id = registry.log_run(run_name="exp", params={}, metrics={"mae": 1.0}, model=model)

    loader = MLflowModelLoader(tracking_uri=tracking_uri, experiment_name="test")
    loaded = loader.load(run_id)

    np.testing.assert_allclose(model.predict(X), loaded.predict(X))


def test_find_latest_run_id_returns_most_recent_matching_prefix(tmp_path: Path) -> None:
    tracking_uri = f"file:{tmp_path / 'mlruns'}"
    registry = MLflowModelRegistry(tracking_uri=tracking_uri, experiment_name="test")
    model, _ = _make_trained_model()

    registry.log_run(run_name="exp_a", params={}, metrics={"mae": 1.0}, model=model)
    rid_b = registry.log_run(run_name="exp_b", params={}, metrics={"mae": 2.0}, model=model)
    registry.log_run(run_name="other", params={}, metrics={"mae": 3.0}, model=model)

    loader = MLflowModelLoader(tracking_uri=tracking_uri, experiment_name="test")

    # The latest run with name starting with "exp" is exp_b
    assert loader.find_latest_run_id("exp") == rid_b


def test_find_latest_run_id_raises_when_no_match(tmp_path: Path) -> None:
    tracking_uri = f"file:{tmp_path / 'mlruns'}"
    registry = MLflowModelRegistry(tracking_uri=tracking_uri, experiment_name="test")
    model, _ = _make_trained_model()
    registry.log_run(run_name="other", params={}, metrics={"mae": 1.0}, model=model)

    loader = MLflowModelLoader(tracking_uri=tracking_uri, experiment_name="test")

    with pytest.raises(FileNotFoundError, match="no runs"):
        loader.find_latest_run_id("missing-prefix")
