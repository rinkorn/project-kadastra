from pathlib import Path

import polars as pl
import pytest

from kadastra.adapters.parquet_coverage_store import ParquetCoverageStore


def test_save_writes_partitioned_parquet_per_resolution(tmp_path: Path) -> None:
    store = ParquetCoverageStore(tmp_path)
    cells = [("h_a", 8), ("h_b", 10)]

    store.save("RU-TA", cells)

    p8 = tmp_path / "region=RU-TA" / "resolution=8" / "data.parquet"
    p10 = tmp_path / "region=RU-TA" / "resolution=10" / "data.parquet"
    assert p8.exists()
    assert p10.exists()

    df8 = pl.read_parquet(p8)
    assert df8.columns == ["h3_index"]
    assert df8["h3_index"].to_list() == ["h_a"]


def test_save_groups_multiple_cells_per_resolution(tmp_path: Path) -> None:
    store = ParquetCoverageStore(tmp_path)
    cells = [("h_a", 8), ("h_b", 8), ("h_c", 10)]

    store.save("RU-TA", cells)

    df = pl.read_parquet(tmp_path / "region=RU-TA" / "resolution=8" / "data.parquet")
    assert df["h3_index"].to_list() == ["h_a", "h_b"]


def test_save_writes_indices_in_sorted_order_for_determinism(tmp_path: Path) -> None:
    store = ParquetCoverageStore(tmp_path)
    cells = [("h_c", 8), ("h_a", 8), ("h_b", 8)]

    store.save("RU-TA", cells)

    df = pl.read_parquet(tmp_path / "region=RU-TA" / "resolution=8" / "data.parquet")
    assert df["h3_index"].to_list() == ["h_a", "h_b", "h_c"]


def test_save_overwrites_previous_partition(tmp_path: Path) -> None:
    store = ParquetCoverageStore(tmp_path)

    store.save("RU-TA", [("h_a", 8)])
    store.save("RU-TA", [("h_b", 8)])

    df = pl.read_parquet(tmp_path / "region=RU-TA" / "resolution=8" / "data.parquet")
    assert df["h3_index"].to_list() == ["h_b"]


def test_save_with_no_cells_creates_no_files(tmp_path: Path) -> None:
    store = ParquetCoverageStore(tmp_path)

    store.save("RU-TA", [])

    assert list(tmp_path.iterdir()) == []


def test_load_returns_dataframe_with_h3_index_and_resolution(tmp_path: Path) -> None:
    store = ParquetCoverageStore(tmp_path)
    store.save("RU-TA", [("h_a", 8), ("h_b", 8), ("h_c", 10)])

    df = store.load("RU-TA", 8)

    assert set(df.columns) == {"h3_index", "resolution"}
    assert sorted(df["h3_index"].to_list()) == ["h_a", "h_b"]
    assert df["resolution"].unique().to_list() == [8]


def test_load_for_missing_resolution_raises(tmp_path: Path) -> None:
    store = ParquetCoverageStore(tmp_path)
    store.save("RU-TA", [("h_a", 8)])

    with pytest.raises(FileNotFoundError):
        store.load("RU-TA", 9)
