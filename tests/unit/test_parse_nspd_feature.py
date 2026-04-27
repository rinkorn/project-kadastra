"""Unit tests for parse_nspd_*_feature pure functions.

These take a single GeoJSON feature dict (as it appears on disk in
data/raw/nspd/{buildings,landplots}-kazan/page-NNNN.json) and return a
flat dict ready for polars. Centroid is computed in EPSG:3857 and
projected to WGS84.
"""

from __future__ import annotations

from typing import Any

from kadastra.etl.parse_nspd_feature import (
    parse_nspd_building_feature,
    parse_nspd_landplot_feature,
)

# Real polygon and properties shape, captured from a Kazan parcel.
_KAZAN_PARCEL_POLYGON_3857 = [
    [
        [5468254.60063592, 7516527.11218784],
        [5468211.28848612, 7516474.28105238],
        [5468206.59564388, 7516467.8674153],
        [5468239.49399934, 7516439.21281993],
        [5468258.29577145, 7516460.81267633],
        [5468255.65457574, 7516463.02113489],
        [5468279.96970424, 7516494.21671869],
        [5468275.02198062, 7516496.69489914],
        [5468277.92186703, 7516502.50620012],
        [5468281.13875952, 7516506.43210297],
        [5468254.60063592, 7516527.11218784],
    ]
]


def _building_feature(**options_overrides: Any) -> dict[str, Any]:
    base_options: dict[str, Any] = {
        "cad_num": "16:50:010406:1",
        "purpose": "Жилой дом",
        "build_record_area": 120.5,
        "cost_value_rub": None,
        "cost_value": 5_400_000.0,
        "cost_index": 44_812.45,
        "year_built": "1985",
        "floors": "5",
        "underground_floors": "1",
        "materials": "Кирпичные",
        "ownership_type": "Частная",
        "build_record_registration_date": "1990-01-15",
        "readable_address": "Республика Татарстан, г Казань, ул Тестовая, д 1",
    }
    base_options.update(options_overrides)
    return {
        "id": 12345678,
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": _KAZAN_PARCEL_POLYGON_3857,
            "crs": {"type": "name", "properties": {"name": "EPSG:3857"}},
        },
        "properties": {
            "category": 36369,
            "categoryName": "Здания",
            "descr": base_options["cad_num"],
            "options": base_options,
        },
    }


def _landplot_feature(**options_overrides: Any) -> dict[str, Any]:
    base_options: dict[str, Any] = {
        "cad_num": "16:50:010406:40",
        "specified_area": 1011.09,
        "cost_value": 16_428_454.11,
        "cost_index": 16_248.26,
        "land_record_category_type": "Земли населенных пунктов",
        "land_record_subtype": "Землепользование",
        "ownership_type": "Частная",
        "land_record_reg_date": "2009-02-16",
        "readable_address": "Казань, ул Университетская, 12/23",
    }
    base_options.update(options_overrides)
    return {
        "id": 38385610,
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": _KAZAN_PARCEL_POLYGON_3857,
            "crs": {"type": "name", "properties": {"name": "EPSG:3857"}},
        },
        "properties": {
            "category": 36368,
            "categoryName": "Земельные участки ЕГРН",
            "descr": base_options["cad_num"],
            "options": base_options,
        },
    }


def test_parse_building_basic_fields() -> None:
    row = parse_nspd_building_feature(_building_feature())

    assert row["geom_data_id"] == 12345678
    assert row["cad_num"] == "16:50:010406:1"
    assert row["asset_class"] == "house"
    assert row["purpose"] == "Жилой дом"
    assert row["area_m2"] == 120.5
    assert row["cost_value_rub"] == 5_400_000.0
    assert row["cost_index_rub_per_m2"] == 44_812.45
    assert row["year_built"] == 1985
    assert row["floors"] == 5
    assert row["underground_floors"] == 1
    assert row["materials"] == "Кирпичные"
    assert row["ownership_type"] == "Частная"
    assert row["readable_address"] == "Республика Татарстан, г Казань, ул Тестовая, д 1"
    assert row["polygon_wkt_3857"].startswith("POLYGON ((")


def test_parse_building_centroid_is_in_kazan() -> None:
    row = parse_nspd_building_feature(_building_feature())

    assert 49.10 < row["lon"] < 49.15
    assert 55.78 < row["lat"] < 55.80


def test_parse_building_with_apartment_purpose() -> None:
    row = parse_nspd_building_feature(_building_feature(purpose="Многоквартирный дом"))
    assert row["asset_class"] == "apartment"


def test_parse_building_with_garage_purpose_returns_none_class() -> None:
    row = parse_nspd_building_feature(_building_feature(purpose="Гараж"))
    assert row["asset_class"] is None


def test_parse_building_handles_missing_optional_fields() -> None:
    row = parse_nspd_building_feature(
        _building_feature(
            year_built="",
            floors=None,
            underground_floors="",
            cost_value=None,
            cost_index=None,
            materials="",
        )
    )
    assert row["year_built"] is None
    assert row["floors"] is None
    assert row["underground_floors"] is None
    assert row["cost_value_rub"] is None
    assert row["cost_index_rub_per_m2"] is None
    assert row["materials"] is None


def test_parse_landplot_basic_fields() -> None:
    row = parse_nspd_landplot_feature(_landplot_feature())

    assert row["geom_data_id"] == 38385610
    assert row["cad_num"] == "16:50:010406:40"
    assert row["asset_class"] == "landplot"
    assert row["area_m2"] == 1011.09
    assert row["cost_value_rub"] == 16_428_454.11
    assert row["cost_index_rub_per_m2"] == 16_248.26
    assert row["land_record_category_type"] == "Земли населенных пунктов"
    assert row["land_record_subtype"] == "Землепользование"
    assert row["polygon_wkt_3857"].startswith("POLYGON ((")


def test_parse_landplot_centroid_is_in_kazan() -> None:
    row = parse_nspd_landplot_feature(_landplot_feature())

    assert 49.10 < row["lon"] < 49.15
    assert 55.78 < row["lat"] < 55.80


def test_parse_landplot_handles_missing_optional_fields() -> None:
    row = parse_nspd_landplot_feature(
        _landplot_feature(
            cost_value=None,
            cost_index=None,
            specified_area=None,
            land_record_category_type=None,
        )
    )
    assert row["cost_value_rub"] is None
    assert row["cost_index_rub_per_m2"] is None
    assert row["area_m2"] is None
    assert row["land_record_category_type"] is None


def test_parse_building_multipolygon_geometry() -> None:
    feat = _building_feature()
    # Wrap into MultiPolygon: list of polygons (each a list of rings)
    feat["geometry"] = {
        "type": "MultiPolygon",
        "coordinates": [_KAZAN_PARCEL_POLYGON_3857],
        "crs": feat["geometry"]["crs"],
    }
    row = parse_nspd_building_feature(feat)

    # Centroid of single-polygon multipolygon should still be in Kazan.
    assert 49.10 < row["lon"] < 49.15
    assert 55.78 < row["lat"] < 55.80
    assert row["polygon_wkt_3857"].startswith("MULTIPOLYGON ((")
