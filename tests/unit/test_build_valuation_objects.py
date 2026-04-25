import io
from dataclasses import dataclass

import polars as pl

from kadastra.domain.asset_class import AssetClass
from kadastra.usecases.build_valuation_objects import BuildValuationObjects


_CSV_HEADER = (
    "osm_id,osm_type,lat,lon,building,levels,flats,material,wall,street,"
    "housenumber,postcode,city,start_date,name,energy_class\n"
)


def _csv_row(osm_id: str, osm_type: str, lat: float, lon: float, building: str) -> str:
    return f"{osm_id},{osm_type},{lat},{lon},{building},,,,,,,,,,,\n"


class _FakeRawData:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read_bytes(self, key: str) -> bytes:
        assert key == "buildings.csv"
        return self._payload


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


def _build_csv_payload() -> bytes:
    rows = [
        _csv_row("1", "way", 55.78, 49.12, "apartments"),
        _csv_row("2", "way", 55.79, 49.13, "house"),
        _csv_row("3", "way", 55.80, 49.14, "retail"),
        _csv_row("4", "way", 55.81, 49.15, "yes"),  # not classified, dropped
    ]
    return (_CSV_HEADER + "".join(rows)).encode()


def test_writes_one_partition_per_asset_class() -> None:
    raw = _FakeRawData(_build_csv_payload())
    store = _FakeStore()
    usecase = BuildValuationObjects(
        raw_data=raw, store=store, buildings_key="buildings.csv"
    )

    usecase.execute("RU-KAZAN-AGG", asset_classes=list(AssetClass))

    assert sorted(c.asset_class.value for c in store.calls) == [
        "apartment",
        "commercial",
        "house",
    ]
    for call in store.calls:
        assert call.region_code == "RU-KAZAN-AGG"


def test_each_partition_only_contains_objects_of_that_class() -> None:
    raw = _FakeRawData(_build_csv_payload())
    store = _FakeStore()
    usecase = BuildValuationObjects(
        raw_data=raw, store=store, buildings_key="buildings.csv"
    )

    usecase.execute("RU-KAZAN-AGG", asset_classes=list(AssetClass))

    by_class = {c.asset_class: c.df for c in store.calls}
    assert by_class[AssetClass.APARTMENT]["asset_class"].unique().to_list() == ["apartment"]
    assert by_class[AssetClass.HOUSE]["asset_class"].unique().to_list() == ["house"]
    assert by_class[AssetClass.COMMERCIAL]["asset_class"].unique().to_list() == ["commercial"]


def test_skips_classes_not_requested() -> None:
    raw = _FakeRawData(_build_csv_payload())
    store = _FakeStore()
    usecase = BuildValuationObjects(
        raw_data=raw, store=store, buildings_key="buildings.csv"
    )

    usecase.execute("RU-KAZAN-AGG", asset_classes=[AssetClass.APARTMENT])

    assert len(store.calls) == 1
    assert store.calls[0].asset_class is AssetClass.APARTMENT


def test_writes_empty_partition_when_no_objects_match() -> None:
    payload = (_CSV_HEADER + _csv_row("1", "way", 55.78, 49.12, "apartments")).encode()
    raw = _FakeRawData(payload)
    store = _FakeStore()
    usecase = BuildValuationObjects(
        raw_data=raw, store=store, buildings_key="buildings.csv"
    )

    usecase.execute("RU-KAZAN-AGG", asset_classes=[AssetClass.APARTMENT, AssetClass.HOUSE])

    by_class = {c.asset_class: c.df for c in store.calls}
    assert by_class[AssetClass.APARTMENT].height == 1
    assert by_class[AssetClass.HOUSE].height == 0


def test_buildings_csv_columns_consistent_across_partitions() -> None:
    raw = _FakeRawData(_build_csv_payload())
    store = _FakeStore()
    usecase = BuildValuationObjects(
        raw_data=raw, store=store, buildings_key="buildings.csv"
    )

    usecase.execute("RU-KAZAN-AGG", asset_classes=list(AssetClass))

    expected_cols = {"object_id", "asset_class", "lat", "lon", "levels", "flats"}
    for call in store.calls:
        assert set(call.df.columns) == expected_cols


def _emit(rows: list[str]) -> bytes:
    return (_CSV_HEADER + "".join(rows)).encode()


def test_reads_buildings_csv_through_raw_data_port() -> None:
    payload = _emit([_csv_row("1", "way", 55.78, 49.12, "apartments")])
    raw = _FakeRawData(payload)
    store = _FakeStore()
    usecase = BuildValuationObjects(
        raw_data=raw, store=store, buildings_key="buildings.csv"
    )

    usecase.execute("RU-KAZAN-AGG", asset_classes=[AssetClass.APARTMENT])

    # Ensure the bytes round-tripped through pl.read_csv: we got the row.
    assert store.calls[0].df.height == 1
    assert store.calls[0].df["object_id"][0] == "way/1"
    _ = io  # keep import: tests pass payloads as bytes
