from dataclasses import dataclass

import polars as pl

from kadastra.domain.asset_class import AssetClass
from kadastra.usecases.build_object_synthetic_target import BuildObjectSyntheticTarget


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
}


def _featured(ac: AssetClass, n: int = 2) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "object_id": f"way/{ac.value}-{i}",
                "asset_class": ac.value,
                "lat": 55.78 + 0.001 * i,
                "lon": 49.12,
                "levels": 5,
                "flats": 30,
                "dist_metro_m": 600.0 + 100 * i,
                "dist_entrance_m": 400.0,
                "count_stations_1km": 1,
                "count_entrances_500m": 1,
                "count_apartments_500m": 5,
                "count_houses_500m": 2,
                "count_commercial_500m": 1,
                "road_length_500m": 1500.0,
            }
            for i in range(n)
        ],
        schema=_FEATURE_SCHEMA,
    )


@dataclass
class _StoreCall:
    region_code: str
    asset_class: AssetClass
    df: pl.DataFrame


class _FakeStore:
    def __init__(self, initial: dict[AssetClass, pl.DataFrame]) -> None:
        self._initial = dict(initial)
        self.calls: list[_StoreCall] = []

    def save(
        self, region_code: str, asset_class: AssetClass, df: pl.DataFrame
    ) -> None:
        self.calls.append(_StoreCall(region_code, asset_class, df))

    def load(
        self, region_code: str, asset_class: AssetClass
    ) -> pl.DataFrame:
        assert region_code == "RU-KAZAN-AGG"
        return self._initial[asset_class]


def test_appends_target_column_for_each_class() -> None:
    store = _FakeStore(
        {
            AssetClass.APARTMENT: _featured(AssetClass.APARTMENT),
            AssetClass.HOUSE: _featured(AssetClass.HOUSE),
            AssetClass.COMMERCIAL: _featured(AssetClass.COMMERCIAL),
        }
    )
    usecase = BuildObjectSyntheticTarget(reader=store, store=store, seed=42)

    usecase.execute("RU-KAZAN-AGG", asset_classes=[AssetClass.APARTMENT, AssetClass.HOUSE, AssetClass.COMMERCIAL])

    assert len(store.calls) == 3
    for call in store.calls:
        assert "synthetic_target_rub_per_m2" in call.df.columns
        assert call.df["synthetic_target_rub_per_m2"].null_count() == 0


def test_seed_is_deterministic_across_runs() -> None:
    initial = {AssetClass.APARTMENT: _featured(AssetClass.APARTMENT, n=20)}

    store_a = _FakeStore(initial)
    store_b = _FakeStore(initial)
    BuildObjectSyntheticTarget(reader=store_a, store=store_a, seed=42).execute(
        "RU-KAZAN-AGG", asset_classes=[AssetClass.APARTMENT]
    )
    BuildObjectSyntheticTarget(reader=store_b, store=store_b, seed=42).execute(
        "RU-KAZAN-AGG", asset_classes=[AssetClass.APARTMENT]
    )

    a = store_a.calls[0].df["synthetic_target_rub_per_m2"].to_list()
    b = store_b.calls[0].df["synthetic_target_rub_per_m2"].to_list()
    assert a == b


def test_skips_empty_partition_but_still_writes_with_target_column() -> None:
    empty = pl.DataFrame(schema=_FEATURE_SCHEMA)
    store = _FakeStore({AssetClass.APARTMENT: empty})
    usecase = BuildObjectSyntheticTarget(reader=store, store=store, seed=42)

    usecase.execute("RU-KAZAN-AGG", asset_classes=[AssetClass.APARTMENT])

    assert len(store.calls) == 1
    assert store.calls[0].df.is_empty()
    assert "synthetic_target_rub_per_m2" in store.calls[0].df.columns


def test_only_processes_requested_classes() -> None:
    store = _FakeStore(
        {
            AssetClass.APARTMENT: _featured(AssetClass.APARTMENT),
            AssetClass.HOUSE: _featured(AssetClass.HOUSE),
            AssetClass.COMMERCIAL: _featured(AssetClass.COMMERCIAL),
        }
    )
    usecase = BuildObjectSyntheticTarget(reader=store, store=store, seed=42)

    usecase.execute("RU-KAZAN-AGG", asset_classes=[AssetClass.HOUSE])

    assert len(store.calls) == 1
    assert store.calls[0].asset_class is AssetClass.HOUSE
