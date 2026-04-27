"""Tests for ParquetNspdSilverStore.

The silver store partitions by region + source (buildings|landplots) so
the two parsed NSPD streams can live side-by-side and the file layout
mirrors the rest of the project (gold/features by region+resolution,
gold/valuation_objects by region+asset_class).
"""

from pathlib import Path

import polars as pl
import pytest

from kadastra.adapters.parquet_nspd_silver_store import ParquetNspdSilverStore


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "geom_data_id": [1, 2, 3],
            "cad_num": ["a", "b", "c"],
            "lat": [55.79, 55.80, 55.78],
            "lon": [49.12, 49.13, 49.11],
        }
    )


def test_save_writes_parquet_under_partitioned_path(tmp_path: Path) -> None:
    store = ParquetNspdSilverStore(tmp_path)

    store.save("RU-KAZAN-AGG", "buildings", _frame())

    expected = tmp_path / "region=RU-KAZAN-AGG" / "source=buildings" / "data.parquet"
    assert expected.is_file()


def test_load_reads_back_what_was_saved(tmp_path: Path) -> None:
    store = ParquetNspdSilverStore(tmp_path)
    df = _frame()

    store.save("RU-KAZAN-AGG", "buildings", df)
    loaded = store.load("RU-KAZAN-AGG", "buildings")

    assert loaded.shape == df.shape
    assert sorted(loaded["cad_num"].to_list()) == sorted(df["cad_num"].to_list())


def test_load_raises_when_partition_missing(tmp_path: Path) -> None:
    store = ParquetNspdSilverStore(tmp_path)

    with pytest.raises(FileNotFoundError):
        store.load("RU-KAZAN-AGG", "landplots")


def test_save_creates_parent_directories(tmp_path: Path) -> None:
    store = ParquetNspdSilverStore(tmp_path / "fresh" / "tree")

    store.save("RU-KAZAN-AGG", "landplots", _frame())

    assert (tmp_path / "fresh" / "tree").is_dir()


def test_save_overwrites_existing_partition(tmp_path: Path) -> None:
    store = ParquetNspdSilverStore(tmp_path)

    store.save("RU-KAZAN-AGG", "buildings", _frame())
    smaller = pl.DataFrame(
        {
            "geom_data_id": [99],
            "cad_num": ["z"],
            "lat": [55.0],
            "lon": [49.0],
        }
    )
    store.save("RU-KAZAN-AGG", "buildings", smaller)

    loaded = store.load("RU-KAZAN-AGG", "buildings")
    assert loaded.height == 1
    assert loaded["cad_num"][0] == "z"


def test_buildings_and_landplots_are_separate_partitions(tmp_path: Path) -> None:
    store = ParquetNspdSilverStore(tmp_path)
    buildings_df = _frame()
    landplots_df = pl.DataFrame(
        {
            "geom_data_id": [10],
            "cad_num": ["land-1"],
            "lat": [55.78],
            "lon": [49.10],
        }
    )

    store.save("RU-KAZAN-AGG", "buildings", buildings_df)
    store.save("RU-KAZAN-AGG", "landplots", landplots_df)

    assert store.load("RU-KAZAN-AGG", "buildings").height == 3
    assert store.load("RU-KAZAN-AGG", "landplots").height == 1
