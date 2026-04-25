from pathlib import Path

import polars as pl

from kadastra.adapters.parquet_gold_feature_store import ParquetGoldFeatureStore


def _gold_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "h3_index": ["h_a", "h_b"],
            "resolution": [8, 8],
            "dist_metro_m": [10.0, 100.0],
            "building_count": [2, 0],
            "road_length_m": [120.5, 0.0],
        }
    )


def test_save_writes_parquet_at_hive_path(tmp_path: Path) -> None:
    store = ParquetGoldFeatureStore(tmp_path)

    store.save("RU-TA", 8, _gold_df())

    expected = tmp_path / "region=RU-TA" / "resolution=8" / "data.parquet"
    assert expected.exists()


def test_save_preserves_columns_and_values(tmp_path: Path) -> None:
    store = ParquetGoldFeatureStore(tmp_path)

    store.save("RU-TA", 8, _gold_df())

    df = pl.read_parquet(tmp_path / "region=RU-TA" / "resolution=8" / "data.parquet")
    assert df.columns == ["h3_index", "resolution", "dist_metro_m", "building_count", "road_length_m"]
    assert df["road_length_m"].to_list() == [120.5, 0.0]


def test_save_overwrites_previous_partition(tmp_path: Path) -> None:
    store = ParquetGoldFeatureStore(tmp_path)
    df1 = pl.DataFrame({"h3_index": ["h_a"], "resolution": [8], "x": [1.0]})
    df2 = pl.DataFrame({"h3_index": ["h_b"], "resolution": [8], "x": [2.0]})

    store.save("RU-TA", 8, df1)
    store.save("RU-TA", 8, df2)

    df = pl.read_parquet(tmp_path / "region=RU-TA" / "resolution=8" / "data.parquet")
    assert df["h3_index"].to_list() == ["h_b"]


def test_load_round_trips_saved_features(tmp_path: Path) -> None:
    store = ParquetGoldFeatureStore(tmp_path)
    store.save("RU-TA", 8, _gold_df())

    df = store.load("RU-TA", 8)

    assert df.columns == ["h3_index", "resolution", "dist_metro_m", "building_count", "road_length_m"]
    assert df["road_length_m"].to_list() == [120.5, 0.0]


def test_load_for_missing_resolution_raises(tmp_path: Path) -> None:
    import pytest

    store = ParquetGoldFeatureStore(tmp_path)
    store.save("RU-TA", 8, _gold_df())

    with pytest.raises(FileNotFoundError):
        store.load("RU-TA", 9)


def test_save_isolates_resolutions(tmp_path: Path) -> None:
    store = ParquetGoldFeatureStore(tmp_path)
    df7 = pl.DataFrame({"h3_index": ["h_a"], "resolution": [7], "x": [1.0]})
    df8 = pl.DataFrame({"h3_index": ["h_b"], "resolution": [8], "x": [2.0]})

    store.save("RU-TA", 7, df7)
    store.save("RU-TA", 8, df8)

    p7 = tmp_path / "region=RU-TA" / "resolution=7" / "data.parquet"
    p8 = tmp_path / "region=RU-TA" / "resolution=8" / "data.parquet"
    assert p7.exists()
    assert p8.exists()
