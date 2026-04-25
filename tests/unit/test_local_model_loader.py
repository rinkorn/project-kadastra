from pathlib import Path

import numpy as np
import pytest

from kadastra.adapters.local_model_loader import LocalModelLoader
from kadastra.adapters.local_model_registry import LocalModelRegistry
from kadastra.ml.train import CatBoostParams, train_catboost


def _make_trained_model():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    y = X[:, 0] + 2 * X[:, 1] - X[:, 2]
    model = train_catboost(X, y, CatBoostParams(iterations=10, learning_rate=0.3, depth=3, seed=0))
    return model, X, y


def test_load_returns_model_with_same_predictions(tmp_path: Path) -> None:
    base = tmp_path / "models"
    registry = LocalModelRegistry(base)
    model, X, _ = _make_trained_model()
    run_id = registry.log_run(run_name="exp", params={}, metrics={}, model=model)

    loader = LocalModelLoader(base)
    loaded = loader.load(run_id)

    np.testing.assert_array_equal(model.predict(X), loaded.predict(X))


def test_load_raises_for_missing_run(tmp_path: Path) -> None:
    loader = LocalModelLoader(tmp_path / "models")

    with pytest.raises(FileNotFoundError):
        loader.load("does-not-exist")


def test_find_latest_run_id_returns_lexicographically_largest_match(tmp_path: Path) -> None:
    base = tmp_path / "models"
    base.mkdir()
    # Run dirs use timestamp suffix → lexicographic max == most recent
    (base / "exp_20260101T000000Z").mkdir()
    (base / "exp_20260601T000000Z").mkdir()
    (base / "exp_20260315T000000Z").mkdir()
    (base / "other_20260601T000000Z").mkdir()  # different prefix

    loader = LocalModelLoader(base)

    assert loader.find_latest_run_id("exp") == "exp_20260601T000000Z"


def test_find_latest_run_id_raises_when_no_match(tmp_path: Path) -> None:
    base = tmp_path / "models"
    base.mkdir()
    (base / "other_20260601T000000Z").mkdir()

    loader = LocalModelLoader(base)

    with pytest.raises(FileNotFoundError, match="no runs"):
        loader.find_latest_run_id("exp")


def test_find_latest_run_id_raises_when_dir_missing(tmp_path: Path) -> None:
    loader = LocalModelLoader(tmp_path / "absent")

    with pytest.raises(FileNotFoundError):
        loader.find_latest_run_id("exp")
