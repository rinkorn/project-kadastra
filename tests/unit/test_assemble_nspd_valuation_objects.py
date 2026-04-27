"""Tests for AssembleNspdValuationObjects.

Reads silver-level NSPD frames (per source) and emits one valuation-object
partition per asset_class, in the schema TrainObjectValuationModel already
expects, with the real cost_index pre-populated as synthetic_target_rub_per_m2
(column name kept for backward compatibility — see ADR-0009).
"""

from __future__ import annotations

import polars as pl

from kadastra.domain.asset_class import AssetClass
from kadastra.usecases.assemble_nspd_valuation_objects import (
    AssembleNspdValuationObjects,
)


class _FakeSilverStore:
    def __init__(self, *, buildings: pl.DataFrame, landplots: pl.DataFrame) -> None:
        self._buildings = buildings
        self._landplots = landplots

    def save(self, region_code: str, source: str, df: pl.DataFrame) -> None:
        raise NotImplementedError

    def load(self, region_code: str, source: str) -> pl.DataFrame:
        if source == "buildings":
            return self._buildings
        if source == "landplots":
            return self._landplots
        raise FileNotFoundError(source)


class _FakeValuationObjectStore:
    def __init__(self) -> None:
        self.calls: list[tuple[str, AssetClass, pl.DataFrame]] = []

    def save(self, region_code: str, asset_class: AssetClass, df: pl.DataFrame) -> None:
        self.calls.append((region_code, asset_class, df))


def _silver_buildings() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "geom_data_id": [1, 2, 3, 4, 5],
            "cad_num": ["a", "b", "c", "d", "e"],
            "purpose": ["Жилой дом", "Многоквартирный дом", "Нежилое", "Жилой дом", "Многоквартирный дом"],
            "asset_class": ["house", "apartment", "commercial", "house", "apartment"],
            "lat": [55.79, 55.80, 55.81, 55.82, 55.83],
            "lon": [49.12, 49.13, 49.14, 49.15, 49.16],
            "area_m2": [120.0, 60.0, 200.0, 90.0, 80.0],
            "cost_value_rub": [5_000_000.0, 5_400_000.0, 12_000_000.0, 4_000_000.0, 6_400_000.0],
            "cost_index_rub_per_m2": [40_000.0, 90_000.0, 60_000.0, 44_000.0, 80_000.0],
            "year_built": [1990, 2010, 2005, 1985, 2018],
            "floors": [3, 9, 5, 2, 12],
            "underground_floors": [1, 1, 0, 0, 1],
            "materials": ["Кирпичные", "Панельные", "Кирпичные", "Деревянные", "Монолитные"],
            "ownership_type": ["Частная"] * 5,
            "registration_date": ["2005-01-01"] * 5,
            "readable_address": [f"Казань addr {i}" for i in range(5)],
            "polygon_wkt_3857": ["POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"] * 5,
        },
        schema_overrides={
            "asset_class": pl.Utf8,
        },
    )


def _silver_landplots() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "geom_data_id": [101, 102, 103],
            "cad_num": ["L1", "L2", "L3"],
            "asset_class": ["landplot", "landplot", "landplot"],
            "lat": [55.78, 55.79, 55.80],
            "lon": [49.10, 49.11, 49.12],
            "area_m2": [500.0, 1000.0, 750.0],
            "cost_value_rub": [3_000_000.0, 8_000_000.0, 5_500_000.0],
            "cost_index_rub_per_m2": [6_000.0, 8_000.0, 7_300.0],
            "land_record_category_type": ["Земли населенных пунктов"] * 3,
            "land_record_subtype": ["Землепользование"] * 3,
            "ownership_type": ["Частная"] * 3,
            "registration_date": ["2009-02-16"] * 3,
            "readable_address": [f"Казань уч {i}" for i in range(3)],
            "polygon_wkt_3857": ["POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"] * 3,
        }
    )


def test_writes_one_partition_per_asset_class() -> None:
    silver = _FakeSilverStore(buildings=_silver_buildings(), landplots=_silver_landplots())
    store = _FakeValuationObjectStore()
    usecase = AssembleNspdValuationObjects(silver_store=silver, valuation_object_store=store)

    usecase.execute(
        "RU-KAZAN-AGG",
        asset_classes=list(AssetClass),
    )

    saved_classes = sorted(c.value for _, c, _ in store.calls)
    assert saved_classes == ["apartment", "commercial", "house", "landplot"]


def test_object_id_uses_nspd_prefix() -> None:
    silver = _FakeSilverStore(buildings=_silver_buildings(), landplots=_silver_landplots())
    store = _FakeValuationObjectStore()
    AssembleNspdValuationObjects(silver_store=silver, valuation_object_store=store).execute(
        "RU-KAZAN-AGG",
        asset_classes=[AssetClass.HOUSE, AssetClass.LANDPLOT],
    )

    by_class = {c: df for _, c, df in store.calls}
    house_ids = by_class[AssetClass.HOUSE]["object_id"].to_list()
    landplot_ids = by_class[AssetClass.LANDPLOT]["object_id"].to_list()

    assert all(oid.startswith("nspd-building/") for oid in house_ids)
    assert all(oid.startswith("nspd-landplot/") for oid in landplot_ids)


def test_target_column_holds_real_cost_index() -> None:
    silver = _FakeSilverStore(buildings=_silver_buildings(), landplots=_silver_landplots())
    store = _FakeValuationObjectStore()
    AssembleNspdValuationObjects(silver_store=silver, valuation_object_store=store).execute(
        "RU-KAZAN-AGG", asset_classes=[AssetClass.HOUSE]
    )

    df = store.calls[0][2]
    assert "synthetic_target_rub_per_m2" in df.columns
    # Both house rows have cost_index 40_000 and 44_000
    assert sorted(df["synthetic_target_rub_per_m2"].to_list()) == [40_000.0, 44_000.0]


def test_levels_maps_floors_for_buildings_and_null_for_landplots() -> None:
    silver = _FakeSilverStore(buildings=_silver_buildings(), landplots=_silver_landplots())
    store = _FakeValuationObjectStore()
    AssembleNspdValuationObjects(silver_store=silver, valuation_object_store=store).execute(
        "RU-KAZAN-AGG", asset_classes=[AssetClass.APARTMENT, AssetClass.LANDPLOT]
    )

    by_class = {c: df for _, c, df in store.calls}
    assert sorted(by_class[AssetClass.APARTMENT]["levels"].to_list()) == [9, 12]
    assert by_class[AssetClass.LANDPLOT]["levels"].to_list() == [None, None, None]


def test_drops_rows_with_null_target() -> None:
    buildings = _silver_buildings().with_columns(
        pl.when(pl.col("geom_data_id") == 1)
        .then(None)
        .otherwise(pl.col("cost_index_rub_per_m2"))
        .alias("cost_index_rub_per_m2")
    )
    silver = _FakeSilverStore(buildings=buildings, landplots=_silver_landplots())
    store = _FakeValuationObjectStore()
    AssembleNspdValuationObjects(silver_store=silver, valuation_object_store=store).execute(
        "RU-KAZAN-AGG", asset_classes=[AssetClass.HOUSE]
    )

    df = store.calls[0][2]
    assert df.height == 1


def test_drops_rows_with_null_asset_class() -> None:
    buildings = _silver_buildings().with_columns(
        pl.when(pl.col("geom_data_id") == 1).then(None).otherwise(pl.col("asset_class")).alias("asset_class")
    )
    silver = _FakeSilverStore(buildings=buildings, landplots=_silver_landplots())
    store = _FakeValuationObjectStore()
    AssembleNspdValuationObjects(silver_store=silver, valuation_object_store=store).execute(
        "RU-KAZAN-AGG", asset_classes=list(AssetClass)
    )

    saved = {c.value: df for _, c, df in store.calls}
    # House had 2 rows, dropped 1 with null asset_class
    assert saved["house"].height == 1


def test_drops_rows_with_zero_or_negative_target() -> None:
    """NSPD occasionally emits rows with cost_value_rub=0 or area=0 →
    target=0. They are broken records, surface as |MAPE|>100% outliers
    in the inspector and must be dropped at assemble time."""
    buildings = _silver_buildings().with_columns(
        pl.when(pl.col("geom_data_id") == 1)
        .then(0.0)
        .otherwise(pl.col("cost_index_rub_per_m2"))
        .alias("cost_index_rub_per_m2"),
        pl.when(pl.col("geom_data_id") == 2)
        .then(-100.0)
        .otherwise(pl.col("cost_index_rub_per_m2"))
        .alias("cost_index_rub_per_m2_negative_marker"),
    )
    # Apply both: rows 1 (zero) and 2 (negative) should drop.
    buildings = buildings.with_columns(
        pl.when(pl.col("geom_data_id") == 2)
        .then(-100.0)
        .otherwise(pl.col("cost_index_rub_per_m2"))
        .alias("cost_index_rub_per_m2")
    ).drop("cost_index_rub_per_m2_negative_marker")
    silver = _FakeSilverStore(buildings=buildings, landplots=_silver_landplots())
    store = _FakeValuationObjectStore()
    AssembleNspdValuationObjects(silver_store=silver, valuation_object_store=store).execute(
        "RU-KAZAN-AGG", asset_classes=[AssetClass.HOUSE, AssetClass.APARTMENT]
    )
    saved = {c.value: df for _, c, df in store.calls}
    # House had 2 rows; geom_data_id=1 (zero) and id=4 are houses → only id=4
    # remains. Apartment had 2 rows; geom_data_id=2 (negative) and id=5 →
    # only id=5 remains.
    assert saved["house"].height == 1
    assert saved["apartment"].height == 1


def test_drops_rows_with_zero_cost_value() -> None:
    """``cost_value_rub == 0`` is a separate broken-record signal: the
    cost is missing in NSPD but the area+target field is non-null
    (computed from a stale cache)."""
    buildings = _silver_buildings().with_columns(
        pl.when(pl.col("geom_data_id") == 1).then(0.0).otherwise(pl.col("cost_value_rub")).alias("cost_value_rub")
    )
    silver = _FakeSilverStore(buildings=buildings, landplots=_silver_landplots())
    store = _FakeValuationObjectStore()
    AssembleNspdValuationObjects(silver_store=silver, valuation_object_store=store).execute(
        "RU-KAZAN-AGG", asset_classes=[AssetClass.HOUSE]
    )
    df = store.calls[0][2]
    assert df.height == 1


def test_dedupes_by_object_id() -> None:
    """NSPD can emit the same record twice (e.g. re-cadastered objects).
    Inspector revealed ~1 200 dup rows in gold for Kazan; the fix is to
    keep the first occurrence of each object_id at assemble time."""
    base = _silver_buildings()
    # Duplicate row 4 (house, geom_data_id=4) by appending a copy.
    duplicated = pl.concat([base, base.filter(pl.col("geom_data_id") == 4)])
    silver = _FakeSilverStore(buildings=duplicated, landplots=_silver_landplots())
    store = _FakeValuationObjectStore()
    AssembleNspdValuationObjects(silver_store=silver, valuation_object_store=store).execute(
        "RU-KAZAN-AGG", asset_classes=[AssetClass.HOUSE]
    )
    df = store.calls[0][2]
    # Originally 2 houses; even after duplicating one, output is still 2.
    assert df.height == 2
    assert df["object_id"].n_unique() == 2


def test_polygon_wkt_3857_passthrough_buildings_and_landplots() -> None:
    """Geometry from silver must propagate to gold for both
    buildings and landplots — the web UI needs object polygons,
    and BuildObjectFeatures runs without selecting columns so the
    only place the geometry can be dropped is here in assemble."""
    silver = _FakeSilverStore(buildings=_silver_buildings(), landplots=_silver_landplots())
    store = _FakeValuationObjectStore()
    AssembleNspdValuationObjects(silver_store=silver, valuation_object_store=store).execute(
        "RU-KAZAN-AGG",
        asset_classes=[AssetClass.HOUSE, AssetClass.LANDPLOT],
    )

    by_class = {c: df for _, c, df in store.calls}
    house_df = by_class[AssetClass.HOUSE]
    landplot_df = by_class[AssetClass.LANDPLOT]
    assert "polygon_wkt_3857" in house_df.columns
    assert "polygon_wkt_3857" in landplot_df.columns
    assert all(wkt is not None and wkt.startswith("POLYGON") for wkt in house_df["polygon_wkt_3857"].to_list())
    assert all(wkt is not None and wkt.startswith("POLYGON") for wkt in landplot_df["polygon_wkt_3857"].to_list())


def test_landplots_partition_skipped_when_landplot_not_requested() -> None:
    silver = _FakeSilverStore(buildings=_silver_buildings(), landplots=_silver_landplots())
    store = _FakeValuationObjectStore()
    AssembleNspdValuationObjects(silver_store=silver, valuation_object_store=store).execute(
        "RU-KAZAN-AGG",
        asset_classes=[AssetClass.HOUSE, AssetClass.APARTMENT, AssetClass.COMMERCIAL],
    )

    saved_classes = {c for _, c, _ in store.calls}
    assert AssetClass.LANDPLOT not in saved_classes
