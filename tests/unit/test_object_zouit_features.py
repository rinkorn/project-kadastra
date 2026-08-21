"""Unit tests for the ЗОУИТ feature block (ADR-0025, п. 3).

Zone fixtures are WKT squares in EPSG:3857 (the НСПД dump's CRS),
built around real Kazan coordinates via pyproj so containment
expectations stay readable.
"""

from __future__ import annotations

from typing import Any

import polars as pl
from pyproj import Transformer

from kadastra.etl.object_zouit_features import (
    ZOUIT_FEATURE_COLUMNS,
    ZOUIT_ZONE_SCHEMA,
    categorize_zouit_type,
    compute_object_zouit_features,
    join_zouit_features,
    parse_zouit_feature,
)

_TO_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

# Test object anchor (Казань, центр) and a far point.
OBJ_LAT, OBJ_LON = 55.7900, 49.1000
FAR_LAT, FAR_LON = 55.8500, 49.2000


def _square_wkt_3857(lat: float, lon: float, half_size_m: float) -> str:
    x, y = _TO_3857.transform(lon, lat)
    h = half_size_m
    return f"POLYGON (({x - h} {y - h}, {x + h} {y - h}, {x + h} {y + h}, {x - h} {y + h}, {x - h} {y - h}))"


def _zone_row(zouit_id: str, type_zone: str, category: str, wkt: str) -> dict[str, object]:
    return {
        "zouit_id": zouit_id,
        "type_zone": type_zone,
        "category": category,
        "geometry_wkt_3857": wkt,
    }


def _zones(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=ZOUIT_ZONE_SCHEMA)


def _objects(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={"object_id": pl.Utf8, "lat": pl.Float64, "lon": pl.Float64},
    )


# ---------------------------------------------------------- categorize


def test_categorize_water_protection() -> None:
    assert categorize_zouit_type("Водоохранная зона") == "water_protection"
    assert categorize_zouit_type("Прибрежная защитная полоса") == "water_protection"
    assert categorize_zouit_type("Зоны затопления и подтопления") == "water_protection"


def test_categorize_sanitary() -> None:
    assert categorize_zouit_type("Санитарно-защитная зона предприятий, сооружений и иных объектов") == "sanitary"
    assert categorize_zouit_type("Зона санитарной охраны источников водоснабжения") == "sanitary"
    assert categorize_zouit_type("Санитарный разрыв (санитарная полоса отчуждения)") == "sanitary"


def test_categorize_heritage_buffer() -> None:
    assert categorize_zouit_type("Зоны охраны объектов культурного наследия") == "heritage_buffer"
    assert categorize_zouit_type("Зона охраны объекта культурного наследия") == "heritage_buffer"
    assert categorize_zouit_type("Защитная зона объекта культурного наследия") == "heritage_buffer"
    assert categorize_zouit_type("Территория объекта культурного наследия") == "heritage_buffer"


def test_categorize_power_line() -> None:
    assert (
        categorize_zouit_type(
            "Охранная зона объектов электроэнергетики (объектов электросетевого хозяйства "
            "и объектов по производству электрической энергии)"
        )
        == "power_line"
    )
    assert categorize_zouit_type("Охранная зона гидроэнергетического объекта") == "power_line"


def test_categorize_pipeline() -> None:
    assert (
        categorize_zouit_type(
            "Охранная зона трубопроводов (газопроводов, нефтепроводов и нефтепродуктопроводов, аммиакопроводов)"
        )
        == "pipeline"
    )
    assert (
        categorize_zouit_type(
            "Зона минимальных расстояний до магистральных или промышленных трубопроводов (газопроводов)"
        )
        == "pipeline"
    )


def test_categorize_aerodrome() -> None:
    assert categorize_zouit_type("Приаэродромная территория") == "aerodrome"


def test_categorize_other_fallback() -> None:
    assert categorize_zouit_type("Охранная зона инженерных коммуникаций") == "other"
    assert categorize_zouit_type("Зона публичного сервитута") == "other"
    assert categorize_zouit_type("Охранная зона геодезического пункта") == "other"
    assert categorize_zouit_type("") == "other"
    assert categorize_zouit_type(None) == "other"


# -------------------------------------------------------------- parse


def _sample_nspd_feature(**option_overrides: Any) -> dict[str, Any]:
    options: dict[str, Any] = {
        "type_zone": "Водоохранная зона",
        "reg_numb_border": "16:16-6.2645",
    }
    options.update(option_overrides)
    return {
        "id": 62833364,
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[5538069.3, 7541628.9], [5538068.6, 7541634.1], [5538069.3, 7541628.9]]],
            "crs": {"type": "name", "properties": {"name": "EPSG:3857"}},
        },
        "properties": {"externalKey": "16:16-6.2645", "options": options},
    }


def test_parse_zouit_feature_happy_path() -> None:
    row = parse_zouit_feature(_sample_nspd_feature())
    assert row is not None
    assert row["zouit_id"] == "16:16-6.2645"
    assert row["type_zone"] == "Водоохранная зона"
    assert row["category"] == "water_protection"
    wkt = row["geometry_wkt_3857"]
    assert isinstance(wkt, str) and wkt.startswith("POLYGON")


def test_parse_zouit_feature_without_geometry_returns_none() -> None:
    feature = _sample_nspd_feature()
    feature["geometry"] = None
    assert parse_zouit_feature(feature) is None


def test_parse_zouit_feature_missing_type_zone_maps_other() -> None:
    feature = _sample_nspd_feature()
    feature["properties"]["options"] = {}
    row = parse_zouit_feature(feature)
    assert row is not None
    assert row["type_zone"] is None
    assert row["category"] == "other"


# ------------------------------------------------------------ compute


def _two_zone_layer() -> pl.DataFrame:
    """A water-protection square and a power-line square, both covering
    the test object; nothing covers the far object."""
    return _zones(
        [
            _zone_row("z1", "Водоохранная зона", "water_protection", _square_wkt_3857(OBJ_LAT, OBJ_LON, 100.0)),
            _zone_row(
                "z2",
                "Охранная зона объектов электроэнергетики",
                "power_line",
                _square_wkt_3857(OBJ_LAT, OBJ_LON, 200.0),
            ),
        ]
    )


def test_object_inside_two_zones() -> None:
    objects = _objects([{"object_id": "a", "lat": OBJ_LAT, "lon": OBJ_LON}])
    df = compute_object_zouit_features(objects, zones=_two_zone_layer())
    row = df.row(0, named=True)
    assert row["object_id"] == "a"
    assert row["inside_zouit"] == 1
    assert row["zouit_types"] == "power_line;water_protection"  # sorted, ;-joined
    assert row["inside_water_protection"] == 1


def test_object_outside_all_zones() -> None:
    objects = _objects([{"object_id": "b", "lat": FAR_LAT, "lon": FAR_LON}])
    df = compute_object_zouit_features(objects, zones=_two_zone_layer())
    row = df.row(0, named=True)
    assert row["inside_zouit"] == 0
    assert row["zouit_types"] is None
    assert row["inside_water_protection"] == 0


def test_inside_zouit_without_water_protection() -> None:
    zones = _zones(
        [
            _zone_row(
                "z1",
                "Охранная зона объектов электроэнергетики",
                "power_line",
                _square_wkt_3857(OBJ_LAT, OBJ_LON, 100.0),
            )
        ]
    )
    objects = _objects([{"object_id": "a", "lat": OBJ_LAT, "lon": OBJ_LON}])
    df = compute_object_zouit_features(objects, zones=zones)
    row = df.row(0, named=True)
    assert row["inside_zouit"] == 1
    assert row["zouit_types"] == "power_line"
    assert row["inside_water_protection"] == 0


def test_null_coords_get_null_features() -> None:
    objects = _objects([{"object_id": "a", "lat": None, "lon": None}])
    df = compute_object_zouit_features(objects, zones=_two_zone_layer())
    row = df.row(0, named=True)
    for col in ZOUIT_FEATURE_COLUMNS:
        assert row[col] is None


def test_empty_zone_layer_gives_null_features() -> None:
    objects = _objects([{"object_id": "a", "lat": OBJ_LAT, "lon": OBJ_LON}])
    df = compute_object_zouit_features(objects, zones=_zones([]))
    for col in ZOUIT_FEATURE_COLUMNS:
        assert df[col][0] is None


def test_empty_objects_frame_schema_stable() -> None:
    df = compute_object_zouit_features(_objects([]), zones=_two_zone_layer())
    assert df.height == 0
    assert df.schema["inside_zouit"] == pl.Int64
    assert df.schema["zouit_types"] == pl.Utf8


# --------------------------------------------------------------- join


def _zouit_features_frame() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "object_id": "way/apartment-1",
                "inside_zouit": 1,
                "zouit_types": "water_protection",
                "inside_water_protection": 1,
            }
        ],
        schema={
            "object_id": pl.Utf8,
            "inside_zouit": pl.Int64,
            "zouit_types": pl.Utf8,
            "inside_water_protection": pl.Int64,
        },
    )


def test_join_appends_columns_and_nulls_for_missing() -> None:
    objects = _objects(
        [
            {"object_id": "way/apartment-1", "lat": OBJ_LAT, "lon": OBJ_LON},
            {"object_id": "way/apartment-2", "lat": FAR_LAT, "lon": FAR_LON},
        ]
    )
    df = join_zouit_features(objects, _zouit_features_frame())
    rows = {r["object_id"]: r for r in df.iter_rows(named=True)}
    assert rows["way/apartment-1"]["inside_zouit"] == 1
    assert rows["way/apartment-1"]["zouit_types"] == "water_protection"
    assert rows["way/apartment-2"]["inside_zouit"] is None
    assert rows["way/apartment-2"]["zouit_types"] is None


def test_join_is_idempotent() -> None:
    objects = join_zouit_features(
        _objects([{"object_id": "way/apartment-1", "lat": OBJ_LAT, "lon": OBJ_LON}]),
        _zouit_features_frame(),
    )
    # Second join over an empty table must replace, not duplicate.
    df = join_zouit_features(
        objects,
        pl.DataFrame(
            schema={
                "object_id": pl.Utf8,
                "inside_zouit": pl.Int64,
                "zouit_types": pl.Utf8,
                "inside_water_protection": pl.Int64,
            }
        ),
    )
    assert df.columns.count("inside_zouit") == 1
    assert df["inside_zouit"][0] is None
