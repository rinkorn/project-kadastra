from pathlib import Path

import polars as pl
import pytest

from kadastra.adapters.parquet_valuation_object_store import ParquetValuationObjectStore
from kadastra.domain.asset_class import AssetClass


def _objects() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "object_id": ["way/1", "way/2"],
            "asset_class": ["apartment", "apartment"],
            "lat": [55.78, 55.79],
            "lon": [49.12, 49.13],
            "levels": [9, 5],
            "flats": [72, 30],
        }
    )


def test_save_writes_parquet_under_partitioned_path(tmp_path: Path) -> None:
    store = ParquetValuationObjectStore(tmp_path)

    store.save("RU-KAZAN-AGG", AssetClass.APARTMENT, _objects())

    expected = (
        tmp_path
        / "region=RU-KAZAN-AGG"
        / "asset_class=apartment"
        / "data.parquet"
    )
    assert expected.is_file()


def test_load_reads_back_what_was_saved(tmp_path: Path) -> None:
    store = ParquetValuationObjectStore(tmp_path)
    df = _objects()

    store.save("RU-KAZAN-AGG", AssetClass.APARTMENT, df)
    loaded = store.load("RU-KAZAN-AGG", AssetClass.APARTMENT)

    assert loaded.shape == df.shape
    assert sorted(loaded["object_id"].to_list()) == sorted(df["object_id"].to_list())


def test_load_raises_when_partition_missing(tmp_path: Path) -> None:
    store = ParquetValuationObjectStore(tmp_path)

    with pytest.raises(FileNotFoundError):
        store.load("RU-KAZAN-AGG", AssetClass.HOUSE)


def test_save_creates_parent_directories(tmp_path: Path) -> None:
    store = ParquetValuationObjectStore(tmp_path / "fresh" / "tree")

    store.save("RU-KAZAN-AGG", AssetClass.COMMERCIAL, _objects())

    assert (tmp_path / "fresh" / "tree").is_dir()


def test_save_overwrites_existing_partition(tmp_path: Path) -> None:
    store = ParquetValuationObjectStore(tmp_path)

    store.save("RU-KAZAN-AGG", AssetClass.APARTMENT, _objects())
    smaller = pl.DataFrame(
        {
            "object_id": ["way/9"],
            "asset_class": ["apartment"],
            "lat": [55.0],
            "lon": [49.0],
            "levels": [None],
            "flats": [None],
        },
        schema={
            "object_id": pl.Utf8,
            "asset_class": pl.Utf8,
            "lat": pl.Float64,
            "lon": pl.Float64,
            "levels": pl.Int64,
            "flats": pl.Int64,
        },
    )
    store.save("RU-KAZAN-AGG", AssetClass.APARTMENT, smaller)

    loaded = store.load("RU-KAZAN-AGG", AssetClass.APARTMENT)
    assert loaded.height == 1
    assert loaded["object_id"][0] == "way/9"
