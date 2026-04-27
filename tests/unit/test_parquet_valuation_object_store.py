from pathlib import Path

import polars as pl
import pytest

from kadastra.adapters.parquet_valuation_object_store import (
    ParquetValuationObjectStore,
    ValuationObjectSchemaError,
)
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

    expected = tmp_path / "region=RU-KAZAN-AGG" / "asset_class=apartment" / "data.parquet"
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


def test_save_rejects_frame_missing_required_column(tmp_path: Path) -> None:
    """Without object_id every downstream consumer breaks. Catch at the
    point of write so the bad parquet never lands on disk."""
    store = ParquetValuationObjectStore(tmp_path)
    bad = pl.DataFrame(
        {
            "asset_class": ["apartment"],
            "lat": [55.78],
            "lon": [49.12],
        }
    )
    with pytest.raises(ValuationObjectSchemaError, match="object_id"):
        store.save("RU-KAZAN-AGG", AssetClass.APARTMENT, bad)


def test_save_rejects_wrong_dtype(tmp_path: Path) -> None:
    """If lat/lon ever drift to Float32 (or string), distance maths and
    KD-tree lookups silently lose precision. Lock the dtype here."""
    store = ParquetValuationObjectStore(tmp_path)
    bad = pl.DataFrame(
        {
            "object_id": ["way/1"],
            "asset_class": ["apartment"],
            "lat": [55.78],
            "lon": [49.12],
        },
        schema={
            "object_id": pl.String,
            "asset_class": pl.String,
            "lat": pl.Float32,
            "lon": pl.Float32,
        },
    )
    with pytest.raises(ValuationObjectSchemaError, match="lat"):
        store.save("RU-KAZAN-AGG", AssetClass.APARTMENT, bad)


def test_load_rejects_existing_parquet_with_bad_schema(tmp_path: Path) -> None:
    """A parquet written outside the store (e.g. a manual fix-up or an
    older pipeline version) must not silently slip through on load —
    the consumer should fail at the boundary, not 50 lines deeper."""
    partition = tmp_path / "region=RU-KAZAN-AGG" / "asset_class=apartment"
    partition.mkdir(parents=True)
    pl.DataFrame(
        {
            "object_id": ["way/1"],
            "asset_class": ["apartment"],
            # Missing lat/lon entirely.
        }
    ).write_parquet(partition / "data.parquet")

    store = ParquetValuationObjectStore(tmp_path)
    with pytest.raises(ValuationObjectSchemaError, match=r"lat|lon"):
        store.load("RU-KAZAN-AGG", AssetClass.APARTMENT)


def test_save_accepts_extra_columns(tmp_path: Path) -> None:
    """The pipeline is additive — feature builders append columns
    stage by stage. Extras must pass without complaint."""
    store = ParquetValuationObjectStore(tmp_path)
    enriched = pl.DataFrame(
        {
            "object_id": ["way/1"],
            "asset_class": ["apartment"],
            "lat": [55.78],
            "lon": [49.12],
            "year_built": [2010],
            "mean_dist_to_school_m": [120.0],
            "polygon_wkt_3857": ["POLYGON((1 2, 3 4, 5 6, 1 2))"],
        }
    )
    store.save("RU-KAZAN-AGG", AssetClass.APARTMENT, enriched)
    loaded = store.load("RU-KAZAN-AGG", AssetClass.APARTMENT)
    assert "year_built" in loaded.columns
    assert "mean_dist_to_school_m" in loaded.columns


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
