"""Tests for GetCellTsorf — Слой 1 cell ЦОФ for the map UI's «ЦОФ-сетка» mode.

Reads ParquetFeatureStore partitions (feature_set=/resolution=) and
returns ``[{"hex", "value"}]`` for one feature column. Mirrors the
GetHexAggregates test pattern but against the feature-store layout.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from kadastra.adapters.parquet_feature_store import ParquetFeatureStore
from kadastra.usecases.get_cell_tsorf import (
    CELL_TSORF_FEATURE_SETS,
    GetCellTsorf,
)


def _write_feature_set(
    store_path: Path,
    region: str,
    resolution: int,
    feature_set: str,
    rows: list[dict[str, object]],
) -> None:
    part = store_path / f"region={region}" / f"feature_set={feature_set}" / f"resolution={resolution}"
    part.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(part / "data.parquet")


def test_execute_returns_hex_value_pairs(tmp_path: Path) -> None:
    base = tmp_path / "features"
    _write_feature_set(
        base,
        "RU-KAZAN-AGG",
        10,
        "walk_dist",
        [
            {"h3_index": "8a", "resolution": 10, "walk_dist_to_school_m": 538.0},
            {"h3_index": "8b", "resolution": 10, "walk_dist_to_school_m": None},
            {"h3_index": "8c", "resolution": 10, "walk_dist_to_school_m": 1200.0},
        ],
    )
    usecase = GetCellTsorf(ParquetFeatureStore(base))

    out = usecase.execute("RU-KAZAN-AGG", 10, "walk_dist", "walk_dist_to_school_m")

    # null row dropped; the other two kept as {hex, value}.
    assert out == [
        {"hex": "8a", "value": 538.0},
        {"hex": "8c", "value": 1200.0},
    ]


def test_execute_unknown_feature_raises_keyerror(tmp_path: Path) -> None:
    base = tmp_path / "features"
    _write_feature_set(
        base,
        "RU-KAZAN-AGG",
        10,
        "walk_dist",
        [{"h3_index": "8a", "resolution": 10, "walk_dist_to_school_m": 538.0}],
    )
    usecase = GetCellTsorf(ParquetFeatureStore(base))

    with pytest.raises(KeyError, match="nope"):
        usecase.execute("RU-KAZAN-AGG", 10, "walk_dist", "nope")


def test_execute_missing_feature_set_raises_filenotfound(tmp_path: Path) -> None:
    usecase = GetCellTsorf(ParquetFeatureStore(tmp_path / "features"))

    with pytest.raises(FileNotFoundError):
        usecase.execute("RU-KAZAN-AGG", 10, "walk_dist", "walk_dist_to_school_m")


def test_list_features_excludes_bookkeeping_columns(tmp_path: Path) -> None:
    base = tmp_path / "features"
    _write_feature_set(
        base,
        "RU-KAZAN-AGG",
        10,
        "metro",
        [
            {
                "h3_index": "8a",
                "resolution": 10,
                "dist_metro_m": 500.0,
                "dist_entrance_m": 400.0,
                "count_stations_1km": 2,
                "count_entrances_500m": 1,
            }
        ],
    )
    usecase = GetCellTsorf(ParquetFeatureStore(base))

    feats = usecase.list_features("RU-KAZAN-AGG", 10, "metro")

    assert set(feats) == {"dist_metro_m", "dist_entrance_m", "count_stations_1km", "count_entrances_500m"}
    assert "h3_index" not in feats and "resolution" not in feats


def test_list_features_missing_set_returns_empty(tmp_path: Path) -> None:
    usecase = GetCellTsorf(ParquetFeatureStore(tmp_path / "features"))

    assert usecase.list_features("RU-KAZAN-AGG", 10, "walk_dist") == []


def test_feature_set_map_covers_all_sets_and_skips_missing(tmp_path: Path) -> None:
    base = tmp_path / "features"
    # Only build one set; the rest should map to [] (not error).
    _write_feature_set(
        base,
        "RU-KAZAN-AGG",
        10,
        "road_density",
        [{"h3_index": "8a", "resolution": 10, "road_length_500m": 950.0}],
    )
    usecase = GetCellTsorf(ParquetFeatureStore(base))

    m = usecase.feature_set_map("RU-KAZAN-AGG", 10)

    # Every documented set is a key, even unbuilt ones.
    assert set(m.keys()) == set(CELL_TSORF_FEATURE_SETS)
    assert m["road_density"] == ["road_length_500m"]
    assert m["walk_dist"] == []


def test_get_cell_detail_returns_features_across_sets(tmp_path: Path) -> None:
    base = tmp_path / "features"
    _write_feature_set(
        base,
        "RU-KAZAN-AGG",
        10,
        "walk_dist",
        [
            {"h3_index": "8a", "resolution": 10, "walk_dist_to_school_m": 538.0},
            {"h3_index": "8b", "resolution": 10, "walk_dist_to_school_m": 1200.0},
        ],
    )
    _write_feature_set(
        base,
        "RU-KAZAN-AGG",
        10,
        "road_density",
        [{"h3_index": "8a", "resolution": 10, "road_length_500m": 950.0}],
    )
    usecase = GetCellTsorf(ParquetFeatureStore(base))

    detail = usecase.get_cell_detail("RU-KAZAN-AGG", 10, "8a")

    assert detail == {
        "walk_dist": {"walk_dist_to_school_m": 538.0},
        "road_density": {"road_length_500m": 950.0},
    }


def test_get_cell_detail_excludes_bookkeeping_columns(tmp_path: Path) -> None:
    base = tmp_path / "features"
    _write_feature_set(
        base,
        "RU-KAZAN-AGG",
        10,
        "metro",
        [{"h3_index": "8a", "resolution": 10, "dist_metro_m": 500.0}],
    )
    usecase = GetCellTsorf(ParquetFeatureStore(base))

    detail = usecase.get_cell_detail("RU-KAZAN-AGG", 10, "8a")

    assert detail == {"metro": {"dist_metro_m": 500.0}}
    assert "h3_index" not in detail["metro"]
    assert "resolution" not in detail["metro"]


def test_get_cell_detail_skips_sets_where_cell_is_absent(tmp_path: Path) -> None:
    base = tmp_path / "features"
    _write_feature_set(
        base,
        "RU-KAZAN-AGG",
        10,
        "walk_dist",
        [{"h3_index": "8a", "resolution": 10, "walk_dist_to_school_m": 538.0}],
    )
    _write_feature_set(
        base,
        "RU-KAZAN-AGG",
        10,
        "road_density",
        [{"h3_index": "8b", "resolution": 10, "road_length_500m": 950.0}],
    )
    usecase = GetCellTsorf(ParquetFeatureStore(base))

    detail = usecase.get_cell_detail("RU-KAZAN-AGG", 10, "8a")

    assert detail == {"walk_dist": {"walk_dist_to_school_m": 538.0}}


def test_get_cell_detail_skips_missing_partitions(tmp_path: Path) -> None:
    base = tmp_path / "features"
    _write_feature_set(
        base,
        "RU-KAZAN-AGG",
        10,
        "walk_dist",
        [{"h3_index": "8a", "resolution": 10, "walk_dist_to_school_m": 538.0}],
    )
    usecase = GetCellTsorf(ParquetFeatureStore(base))

    detail = usecase.get_cell_detail("RU-KAZAN-AGG", 10, "8a")

    # Only walk_dist built; every other CELL_TSORF_FEATURE_SETS partition
    # is absent and must be silently skipped, not raise.
    assert detail == {"walk_dist": {"walk_dist_to_school_m": 538.0}}


def test_get_cell_detail_returns_empty_when_cell_not_found_anywhere(tmp_path: Path) -> None:
    base = tmp_path / "features"
    _write_feature_set(
        base,
        "RU-KAZAN-AGG",
        10,
        "walk_dist",
        [{"h3_index": "8a", "resolution": 10, "walk_dist_to_school_m": 538.0}],
    )
    usecase = GetCellTsorf(ParquetFeatureStore(base))

    assert usecase.get_cell_detail("RU-KAZAN-AGG", 10, "missing") == {}
