from pathlib import Path

import polars as pl

from kadastra.adapters.parquet_feature_store import ParquetFeatureStore


def _features_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "h3_index": ["h_a", "h_b"],
            "resolution": [8, 8],
            "dist_metro_m": [10.0, 100.0],
            "count_stations_1km": [1, 0],
        }
    )


def test_save_writes_parquet_at_hive_partitioned_path(tmp_path: Path) -> None:
    store = ParquetFeatureStore(tmp_path)

    store.save("RU-TA", 8, "metro", _features_df())

    expected_path = tmp_path / "region=RU-TA" / "feature_set=metro" / "resolution=8" / "data.parquet"
    assert expected_path.exists()


def test_save_preserves_dataframe_schema_and_values(tmp_path: Path) -> None:
    store = ParquetFeatureStore(tmp_path)

    store.save("RU-TA", 8, "metro", _features_df())

    df = pl.read_parquet(
        tmp_path / "region=RU-TA" / "feature_set=metro" / "resolution=8" / "data.parquet"
    )
    assert df.columns == ["h3_index", "resolution", "dist_metro_m", "count_stations_1km"]
    assert df["dist_metro_m"].to_list() == [10.0, 100.0]
    assert df["count_stations_1km"].to_list() == [1, 0]


def test_save_overwrites_previous_partition(tmp_path: Path) -> None:
    store = ParquetFeatureStore(tmp_path)
    df1 = pl.DataFrame({"h3_index": ["h_a"], "resolution": [8], "feat": [1.0]})
    df2 = pl.DataFrame({"h3_index": ["h_b"], "resolution": [8], "feat": [2.0]})

    store.save("RU-TA", 8, "metro", df1)
    store.save("RU-TA", 8, "metro", df2)

    df = pl.read_parquet(
        tmp_path / "region=RU-TA" / "feature_set=metro" / "resolution=8" / "data.parquet"
    )
    assert df["h3_index"].to_list() == ["h_b"]


def test_load_round_trips_saved_features(tmp_path: Path) -> None:
    store = ParquetFeatureStore(tmp_path)
    store.save("RU-TA", 8, "metro", _features_df())

    df = store.load("RU-TA", 8, "metro")

    assert df.columns == ["h3_index", "resolution", "dist_metro_m", "count_stations_1km"]
    assert df["dist_metro_m"].to_list() == [10.0, 100.0]


def test_load_for_missing_feature_set_raises(tmp_path: Path) -> None:
    import pytest

    store = ParquetFeatureStore(tmp_path)
    store.save("RU-TA", 8, "metro", _features_df())

    with pytest.raises(FileNotFoundError):
        store.load("RU-TA", 8, "buildings")


def test_save_isolates_feature_sets(tmp_path: Path) -> None:
    store = ParquetFeatureStore(tmp_path)
    metro = pl.DataFrame({"h3_index": ["h_a"], "resolution": [8], "dist_metro_m": [10.0]})
    roads = pl.DataFrame({"h3_index": ["h_a"], "resolution": [8], "road_length_m": [200.0]})

    store.save("RU-TA", 8, "metro", metro)
    store.save("RU-TA", 8, "roads", roads)

    metro_path = tmp_path / "region=RU-TA" / "feature_set=metro" / "resolution=8" / "data.parquet"
    roads_path = tmp_path / "region=RU-TA" / "feature_set=roads" / "resolution=8" / "data.parquet"
    assert metro_path.exists()
    assert roads_path.exists()
