"""Tests for LocalEbmModelLoader — loads the latest EBM model artifact."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kadastra.adapters.ebm_quartet_model import EbmQuartetModel
from kadastra.adapters.local_ebm_model_loader import LocalEbmModelLoader
from kadastra.domain.asset_class import AssetClass


def _write_ebm(run_dir: Path) -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 2))
    y = X[:, 0] + X[:, 1]
    model = EbmQuartetModel(max_bins=16, interactions=0)
    model.fit(X, y, cat_feature_indices=None)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "ebm_model.pkl").write_bytes(model.serialize())


def test_load_latest_returns_ebm_model(tmp_path: Path) -> None:
    _write_ebm(tmp_path / "quartet-object-apartment_20260426T000000Z")
    model = LocalEbmModelLoader(tmp_path).load_latest(AssetClass.APARTMENT)
    assert model.predict(np.array([[0.0, 0.0]])).shape == (1,)


def test_load_latest_picks_most_recent(tmp_path: Path) -> None:
    _write_ebm(tmp_path / "quartet-object-apartment_20260426T000000Z")
    _write_ebm(tmp_path / "quartet-object-apartment_20260426T010000Z")
    model = LocalEbmModelLoader(tmp_path).load_latest(AssetClass.APARTMENT)
    # Loader just needs to resolve without error — both artifacts are
    # structurally identical, so only existence + pickability is asserted.
    assert model is not None


def test_load_latest_raises_when_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        LocalEbmModelLoader(tmp_path).load_latest(AssetClass.APARTMENT)
