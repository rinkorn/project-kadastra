"""Tests for read_nspd_{buildings,landplots}_dir.

These walk a directory of ``page-NNNN.json`` files (the structure that
``scripts/download_nspd_layer.py`` writes) and assemble a polars
DataFrame with parsed rows. Helper-only files such as ``_state.json``
must be ignored.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from kadastra.etl.read_nspd_dir import (
    read_nspd_buildings_dir,
    read_nspd_landplots_dir,
)

_KAZAN_POLYGON_3857 = [
    [
        [5468254.60063592, 7516527.11218784],
        [5468211.28848612, 7516474.28105238],
        [5468206.59564388, 7516467.8674153],
        [5468239.49399934, 7516439.21281993],
        [5468258.29577145, 7516460.81267633],
        [5468254.60063592, 7516527.11218784],
    ]
]


def _building_payload(features: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "data": {"type": "FeatureCollection", "features": features},
        "meta": [{"totalCount": len(features), "categoryId": 36369}],
    }


def _landplot_payload(features: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "data": {"type": "FeatureCollection", "features": features},
        "meta": [{"totalCount": len(features), "categoryId": 36368}],
    }


def _building_feature(idx: int, *, purpose: str = "Жилой дом") -> dict[str, Any]:
    return {
        "id": 100000 + idx,
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": _KAZAN_POLYGON_3857,
            "crs": {"type": "name", "properties": {"name": "EPSG:3857"}},
        },
        "properties": {
            "category": 36369,
            "descr": f"16:50:010406:{idx}",
            "options": {
                "cad_num": f"16:50:010406:{idx}",
                "purpose": purpose,
                "build_record_area": 100.0 + idx,
                "cost_value": 5_000_000.0 + 1000.0 * idx,
                "cost_index": 40_000.0 + 100.0 * idx,
                "year_built": str(1990 + idx),
                "floors": str(3 + (idx % 5)),
                "materials": "Кирпичные",
                "ownership_type": "Частная",
                "build_record_registration_date": "2010-05-12",
                "readable_address": f"Казань, ул Тестовая, д {idx}",
            },
        },
    }


def _landplot_feature(idx: int) -> dict[str, Any]:
    return {
        "id": 200000 + idx,
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": _KAZAN_POLYGON_3857,
            "crs": {"type": "name", "properties": {"name": "EPSG:3857"}},
        },
        "properties": {
            "category": 36368,
            "descr": f"16:50:010406:{1000 + idx}",
            "options": {
                "cad_num": f"16:50:010406:{1000 + idx}",
                "specified_area": 500.0 + 10.0 * idx,
                "cost_value": 8_000_000.0 + 100_000.0 * idx,
                "cost_index": 16_000.0 + 50.0 * idx,
                "land_record_category_type": "Земли населенных пунктов",
                "land_record_subtype": "Землепользование",
                "ownership_type": "Частная",
                "land_record_reg_date": "2009-02-16",
                "readable_address": f"Казань, ул Тестовая, уч {idx}",
            },
        },
    }


def test_read_buildings_dir_loads_all_pages(tmp_path: Path) -> None:
    page0 = _building_payload([_building_feature(i) for i in range(3)])
    page1 = _building_payload(
        [_building_feature(i, purpose="Многоквартирный дом") for i in range(3, 5)]
    )
    (tmp_path / "page-0000.json").write_text(json.dumps(page0))
    (tmp_path / "page-0001.json").write_text(json.dumps(page1))
    (tmp_path / "_state.json").write_text(json.dumps({"layer_id": 36049}))

    df = read_nspd_buildings_dir(tmp_path)

    assert df.height == 5
    assert sorted(df["geom_data_id"].to_list()) == [100000, 100001, 100002, 100003, 100004]
    assert set(df["asset_class"].to_list()) == {"house", "apartment"}


def test_read_buildings_dir_returns_expected_polars_schema(tmp_path: Path) -> None:
    page = _building_payload([_building_feature(0)])
    (tmp_path / "page-0000.json").write_text(json.dumps(page))

    df = read_nspd_buildings_dir(tmp_path)

    assert df.schema["geom_data_id"] == pl.Int64
    assert df.schema["cad_num"] == pl.Utf8
    assert df.schema["asset_class"] == pl.Utf8
    assert df.schema["lat"] == pl.Float64
    assert df.schema["lon"] == pl.Float64
    assert df.schema["area_m2"] == pl.Float64
    assert df.schema["cost_value_rub"] == pl.Float64
    assert df.schema["cost_index_rub_per_m2"] == pl.Float64
    assert df.schema["year_built"] == pl.Int64
    assert df.schema["floors"] == pl.Int64
    assert df.schema["polygon_wkt_3857"] == pl.Utf8


def test_read_buildings_dir_empty_returns_empty_frame(tmp_path: Path) -> None:
    (tmp_path / "_state.json").write_text("{}")

    df = read_nspd_buildings_dir(tmp_path)

    assert df.height == 0
    assert "geom_data_id" in df.columns


def test_read_landplots_dir_loads_all_pages(tmp_path: Path) -> None:
    page0 = _landplot_payload([_landplot_feature(i) for i in range(3)])
    page1 = _landplot_payload([_landplot_feature(i) for i in range(3, 7)])
    (tmp_path / "page-0000.json").write_text(json.dumps(page0))
    (tmp_path / "page-0001.json").write_text(json.dumps(page1))

    df = read_nspd_landplots_dir(tmp_path)

    assert df.height == 7
    assert all(c == "landplot" for c in df["asset_class"].to_list())
    assert df["land_record_category_type"][0] == "Земли населенных пунктов"


def test_read_landplots_dir_schema(tmp_path: Path) -> None:
    page = _landplot_payload([_landplot_feature(0)])
    (tmp_path / "page-0000.json").write_text(json.dumps(page))

    df = read_nspd_landplots_dir(tmp_path)

    assert df.schema["geom_data_id"] == pl.Int64
    assert df.schema["cad_num"] == pl.Utf8
    assert df.schema["asset_class"] == pl.Utf8
    assert df.schema["land_record_category_type"] == pl.Utf8
    assert df.schema["cost_index_rub_per_m2"] == pl.Float64
