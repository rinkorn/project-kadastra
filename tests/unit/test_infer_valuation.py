import h3
import numpy as np
import polars as pl
import pytest
from catboost import CatBoostRegressor

from kadastra.ml.train import CatBoostParams, train_catboost
from kadastra.usecases.infer_valuation import InferValuation

KAZAN_LAT, KAZAN_LON = 55.7887, 49.1221


def _gold(n: int, resolution: int = 8) -> pl.DataFrame:
    cells = [h3.latlng_to_cell(KAZAN_LAT + 0.01 * i, KAZAN_LON, resolution) for i in range(n)]
    return pl.DataFrame(
        {
            "h3_index": cells,
            "resolution": [resolution] * n,
            "feat_a": list(range(n)),
            "feat_b": [i % 3 for i in range(n)],
        }
    )


def _train_simple_model(gold: pl.DataFrame) -> CatBoostRegressor:
    feature_cols = [c for c in gold.columns if c not in {"h3_index", "resolution"}]
    X = gold.select(feature_cols).to_numpy().astype(np.float64)
    y = X[:, 0] * 10.0 + X[:, 1]  # deterministic synthetic
    return train_catboost(X, y, CatBoostParams(iterations=20, learning_rate=0.3, depth=4, seed=42))


class FakeModelLoader:
    def __init__(self, models_by_run_id: dict[str, CatBoostRegressor], latest_id: str) -> None:
        self._models = models_by_run_id
        self._latest = latest_id
        self.find_calls: list[str] = []
        self.load_calls: list[str] = []

    def load(self, run_id: str) -> CatBoostRegressor:
        self.load_calls.append(run_id)
        return self._models[run_id]

    def find_latest_run_id(self, run_name_prefix: str) -> str:
        self.find_calls.append(run_name_prefix)
        return self._latest


class FakeGoldReader:
    def __init__(self, by_resolution: dict[int, pl.DataFrame]) -> None:
        self._by_resolution = by_resolution

    def load(self, region_code: str, resolution: int) -> pl.DataFrame:
        return self._by_resolution[resolution]


class FakePredictionStore:
    def __init__(self) -> None:
        self.saved: list[tuple[str, int, pl.DataFrame]] = []

    def save(self, region_code: str, resolution: int, df: pl.DataFrame) -> None:
        self.saved.append((region_code, resolution, df))


def test_execute_with_explicit_run_id_skips_find_latest() -> None:
    gold = _gold(10)
    model = _train_simple_model(gold)
    loader = FakeModelLoader({"run-x": model}, latest_id="run-y")
    store = FakePredictionStore()

    usecase = InferValuation(
        model_loader=loader,
        gold_reader=FakeGoldReader({8: gold}),
        prediction_store=store,
        run_name_prefix="catboost-baseline-res",
    )

    used = usecase.execute("RU-TA", 8, run_id="run-x")

    assert used == "run-x"
    assert loader.find_calls == []
    assert loader.load_calls == ["run-x"]


def test_execute_without_run_id_uses_find_latest() -> None:
    gold = _gold(10)
    model = _train_simple_model(gold)
    loader = FakeModelLoader({"run-y": model}, latest_id="run-y")
    store = FakePredictionStore()

    usecase = InferValuation(
        model_loader=loader,
        gold_reader=FakeGoldReader({8: gold}),
        prediction_store=store,
        run_name_prefix="catboost-baseline-res",
    )

    used = usecase.execute("RU-TA", 8)

    assert used == "run-y"
    assert loader.find_calls == ["catboost-baseline-res8"]


def test_execute_saves_predictions_for_every_hex() -> None:
    gold = _gold(10)
    model = _train_simple_model(gold)
    loader = FakeModelLoader({"run-y": model}, latest_id="run-y")
    store = FakePredictionStore()

    usecase = InferValuation(
        model_loader=loader,
        gold_reader=FakeGoldReader({8: gold}),
        prediction_store=store,
        run_name_prefix="catboost-baseline-res",
    )
    usecase.execute("RU-TA", 8)

    assert len(store.saved) == 1
    region, resolution, df = store.saved[0]
    assert region == "RU-TA"
    assert resolution == 8
    assert set(df.columns) == {"h3_index", "resolution", "predicted_value"}
    assert df.height == 10


def test_predictions_match_direct_model_predict() -> None:
    gold = _gold(10)
    model = _train_simple_model(gold)
    loader = FakeModelLoader({"run-y": model}, latest_id="run-y")
    store = FakePredictionStore()

    usecase = InferValuation(
        model_loader=loader,
        gold_reader=FakeGoldReader({8: gold}),
        prediction_store=store,
        run_name_prefix="catboost-baseline-res",
    )
    usecase.execute("RU-TA", 8)

    feature_cols = [c for c in gold.columns if c not in {"h3_index", "resolution"}]
    expected = model.predict(gold.select(feature_cols).to_numpy().astype(np.float64))

    saved_df = store.saved[0][2].sort("h3_index")
    expected_df = (
        gold.select(["h3_index"]).with_columns(pl.Series("predicted_value", expected)).sort("h3_index")
    )
    np.testing.assert_allclose(
        saved_df["predicted_value"].to_numpy(),
        expected_df["predicted_value"].to_numpy(),
    )


def test_execute_propagates_loader_not_found() -> None:
    class FailingLoader:
        def load(self, run_id: str) -> CatBoostRegressor:  # noqa: ARG002
            raise FileNotFoundError("nope")

        def find_latest_run_id(self, run_name_prefix: str) -> str:  # noqa: ARG002
            raise FileNotFoundError("no runs")

    usecase = InferValuation(
        model_loader=FailingLoader(),
        gold_reader=FakeGoldReader({8: _gold(5)}),
        prediction_store=FakePredictionStore(),
        run_name_prefix="catboost-baseline-res",
    )

    with pytest.raises(FileNotFoundError):
        usecase.execute("RU-TA", 8)
