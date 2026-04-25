from dataclasses import dataclass

import polars as pl
from catboost import CatBoostRegressor

from kadastra.domain.asset_class import AssetClass
from kadastra.usecases.infer_object_valuation import InferObjectValuation

KAZAN_LAT, KAZAN_LON = 55.7887, 49.1221


_FEATURE_SCHEMA = {
    "object_id": pl.Utf8,
    "asset_class": pl.Utf8,
    "lat": pl.Float64,
    "lon": pl.Float64,
    "levels": pl.Int64,
    "flats": pl.Int64,
    "dist_metro_m": pl.Float64,
    "dist_entrance_m": pl.Float64,
    "count_stations_1km": pl.Int64,
    "count_entrances_500m": pl.Int64,
    "count_apartments_500m": pl.Int64,
    "count_houses_500m": pl.Int64,
    "count_commercial_500m": pl.Int64,
    "road_length_500m": pl.Float64,
    "synthetic_target_rub_per_m2": pl.Float64,
}


def _featured(ac: AssetClass, n: int = 30) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "object_id": f"way/{ac.value}-{i}",
                "asset_class": ac.value,
                "lat": KAZAN_LAT + 0.001 * i,
                "lon": KAZAN_LON,
                "levels": (i % 9) + 1,
                "flats": ((i % 7) + 1) * 10,
                "dist_metro_m": 200.0 + 30.0 * i,
                "dist_entrance_m": 150.0 + 25.0 * i,
                "count_stations_1km": (i % 3),
                "count_entrances_500m": (i % 2),
                "count_apartments_500m": (i % 5),
                "count_houses_500m": (i % 4),
                "count_commercial_500m": (i % 3),
                "road_length_500m": 500.0 + 50.0 * (i % 8),
                "synthetic_target_rub_per_m2": 50_000.0 + 500.0 * i,
            }
            for i in range(n)
        ],
        schema=_FEATURE_SCHEMA,
    )


def _is_numeric(dtype: pl.DataType) -> bool:
    return dtype.is_numeric()


def _is_categorical(dtype: pl.DataType) -> bool:
    return dtype == pl.Utf8 or dtype == pl.Categorical


def _trained_model(df: pl.DataFrame) -> CatBoostRegressor:
    """Mirror TrainObjectValuationModel.execute()'s feature handling so
    the model in the test ends up with the same column ordering and
    cat_features that the production code will use.
    """
    excluded = {
        "object_id",
        "asset_class",
        "lat",
        "lon",
        "synthetic_target_rub_per_m2",
        "cost_value_rub",
    }
    numeric_cols = [
        c
        for c in df.columns
        if c not in excluded and _is_numeric(df.schema[c])
    ]
    categorical_cols = [
        c
        for c in df.columns
        if c not in excluded and _is_categorical(df.schema[c])
    ]
    feature_cols = numeric_cols + categorical_cols
    cat_indices = list(range(len(numeric_cols), len(feature_cols)))

    df = df.with_columns(
        [pl.col(c).fill_null(0).cast(pl.Float64) for c in numeric_cols]
        + [
            pl.col(c).fill_null("__missing__").cast(pl.Utf8)
            for c in categorical_cols
        ]
    )
    X = df.select(feature_cols).to_numpy()
    y = df["synthetic_target_rub_per_m2"].to_numpy()
    model = CatBoostRegressor(
        iterations=20,
        learning_rate=0.3,
        depth=4,
        verbose=False,
        allow_writing_files=False,
        cat_features=cat_indices or None,
    )
    model.fit(X, y, cat_features=cat_indices or None)
    return model


class _FakeReader:
    def __init__(self, by_class: dict[AssetClass, pl.DataFrame]) -> None:
        self._by_class = by_class

    def load(self, region_code: str, asset_class: AssetClass) -> pl.DataFrame:
        return self._by_class[asset_class]


@dataclass
class _StoreCall:
    region_code: str
    asset_class: AssetClass
    df: pl.DataFrame


class _FakeStore:
    def __init__(self) -> None:
        self.calls: list[_StoreCall] = []

    def save(
        self, region_code: str, asset_class: AssetClass, df: pl.DataFrame
    ) -> None:
        self.calls.append(_StoreCall(region_code, asset_class, df))


class _FakeLoader:
    def __init__(self, model: CatBoostRegressor) -> None:
        self._model = model
        self.requested_run_ids: list[str] = []
        self.requested_prefixes: list[str] = []

    def load(self, run_id: str) -> CatBoostRegressor:
        self.requested_run_ids.append(run_id)
        return self._model

    def find_latest_run_id(self, run_name_prefix: str) -> str:
        self.requested_prefixes.append(run_name_prefix)
        return f"{run_name_prefix}-latest"


def test_writes_predictions_per_object_to_store() -> None:
    df = _featured(AssetClass.APARTMENT, 30)
    model = _trained_model(df)
    reader = _FakeReader({AssetClass.APARTMENT: df})
    store = _FakeStore()
    loader = _FakeLoader(model)

    usecase = InferObjectValuation(
        model_loader=loader,
        reader=reader,
        prediction_store=store,
        run_name_prefix="catboost-object-",
    )
    usecase.execute("RU-KAZAN-AGG", AssetClass.APARTMENT)

    assert len(store.calls) == 1
    saved = store.calls[0].df
    assert set(saved.columns) == {"object_id", "asset_class", "lat", "lon", "predicted_value"}
    assert saved.height == df.height
    assert saved["predicted_value"].null_count() == 0


def test_loads_latest_run_when_run_id_not_provided() -> None:
    df = _featured(AssetClass.HOUSE, 30)
    loader = _FakeLoader(_trained_model(df))
    InferObjectValuation(
        model_loader=loader,
        reader=_FakeReader({AssetClass.HOUSE: df}),
        prediction_store=_FakeStore(),
        run_name_prefix="catboost-object-",
    ).execute("RU-KAZAN-AGG", AssetClass.HOUSE)

    assert loader.requested_prefixes == ["catboost-object-house"]


def test_uses_explicit_run_id_when_provided() -> None:
    df = _featured(AssetClass.COMMERCIAL, 30)
    loader = _FakeLoader(_trained_model(df))
    usecase = InferObjectValuation(
        model_loader=loader,
        reader=_FakeReader({AssetClass.COMMERCIAL: df}),
        prediction_store=_FakeStore(),
        run_name_prefix="catboost-object-",
    )

    usecase.execute("RU-KAZAN-AGG", AssetClass.COMMERCIAL, run_id="explicit-42")

    assert loader.requested_run_ids == ["explicit-42"]
    assert loader.requested_prefixes == []


def test_returns_resolved_run_id() -> None:
    df = _featured(AssetClass.APARTMENT, 30)
    loader = _FakeLoader(_trained_model(df))
    usecase = InferObjectValuation(
        model_loader=loader,
        reader=_FakeReader({AssetClass.APARTMENT: df}),
        prediction_store=_FakeStore(),
        run_name_prefix="catboost-object-",
    )

    auto = usecase.execute("RU-KAZAN-AGG", AssetClass.APARTMENT)
    explicit = usecase.execute("RU-KAZAN-AGG", AssetClass.APARTMENT, run_id="x")

    assert auto == "catboost-object-apartment-latest"
    assert explicit == "x"


def test_excludes_cost_value_rub_target_leak() -> None:
    """cost_value_rub is the EГРН total from which cost_index = cost_value
    / area_m2 is derived; must be excluded from features at inference too.
    """
    df = _featured(AssetClass.APARTMENT, 30).with_columns(
        pl.lit(5_000_000.0).alias("cost_value_rub")
    )
    model = _trained_model(df)
    reader = _FakeReader({AssetClass.APARTMENT: df})
    store = _FakeStore()
    loader = _FakeLoader(model)

    InferObjectValuation(
        model_loader=loader,
        reader=reader,
        prediction_store=store,
        run_name_prefix="catboost-object-",
    ).execute("RU-KAZAN-AGG", AssetClass.APARTMENT)

    saved = store.calls[0].df
    assert saved.height == df.height
    assert saved["predicted_value"].null_count() == 0


def test_passes_string_columns_through_to_categorical_model() -> None:
    """Models trained with cat_features expect the same string columns at
    inference time — not dropped, not cast to numeric. Inference must
    pass categorical columns straight through (with a sentinel for nulls).
    """
    df = _featured(AssetClass.HOUSE, 30).with_columns(
        pl.Series(
            "materials",
            ["Кирпичные", "Панельные", "Монолитные"] * 10,
            dtype=pl.Utf8,
        )
    )
    model = _trained_model(df)
    reader = _FakeReader({AssetClass.HOUSE: df})
    store = _FakeStore()
    loader = _FakeLoader(model)

    InferObjectValuation(
        model_loader=loader,
        reader=reader,
        prediction_store=store,
        run_name_prefix="catboost-object-",
    ).execute("RU-KAZAN-AGG", AssetClass.HOUSE)

    saved = store.calls[0].df
    assert saved.height == df.height
    assert saved["predicted_value"].null_count() == 0
