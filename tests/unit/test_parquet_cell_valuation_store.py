"""Tests for ParquetCellValuationStore (ADR-0029)."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from kadastra.adapters.parquet_cell_valuation_store import (
    CellValuationSchemaError,
    ParquetCellValuationStore,
)
from kadastra.domain.asset_class import AssetClass


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "h3_index": ["8a10a800007fff"],
            "lat": [55.79],
            "lon": [49.11],
            "reference_variant": ["default"],
            "reference_rub_per_m2": [100_000.0],
            "location_score_rub_per_m2": [90_000.0],
            "n_sample_objects": [3],
            "sample_covered": [True],
        }
    )


def test_save_load_roundtrip(tmp_path: Path) -> None:
    store = ParquetCellValuationStore(tmp_path)
    store.save("RU-KAZAN-AGG", AssetClass.HOUSE, _frame())
    loaded = store.load("RU-KAZAN-AGG", AssetClass.HOUSE)
    assert loaded.equals(_frame())
    assert (tmp_path / "region=RU-KAZAN-AGG" / "asset_class=house" / "data.parquet").is_file()


def test_save_rejects_missing_columns(tmp_path: Path) -> None:
    store = ParquetCellValuationStore(tmp_path)
    with pytest.raises(CellValuationSchemaError, match="h3_index"):
        store.save("RU-KAZAN-AGG", AssetClass.HOUSE, _frame().drop("h3_index"))


def test_load_missing_partition_raises(tmp_path: Path) -> None:
    store = ParquetCellValuationStore(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.load("RU-KAZAN-AGG", AssetClass.HOUSE)
