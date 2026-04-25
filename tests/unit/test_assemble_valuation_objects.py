import polars as pl

from kadastra.domain.asset_class import AssetClass
from kadastra.etl.valuation_objects import assemble_valuation_objects


def _buildings(rows: list[dict[str, object]]) -> pl.DataFrame:
    schema = {
        "osm_id": pl.Utf8,
        "osm_type": pl.Utf8,
        "lat": pl.Float64,
        "lon": pl.Float64,
        "building": pl.Utf8,
        "levels": pl.Utf8,
        "flats": pl.Utf8,
    }
    return pl.DataFrame(rows, schema=schema)


def test_returns_expected_columns() -> None:
    buildings = _buildings(
        [
            {
                "osm_id": "1",
                "osm_type": "way",
                "lat": 55.78,
                "lon": 49.12,
                "building": "apartments",
                "levels": "9",
                "flats": "72",
            }
        ]
    )

    result = assemble_valuation_objects(buildings)

    assert set(result.columns) == {
        "object_id",
        "asset_class",
        "lat",
        "lon",
        "levels",
        "flats",
    }


def test_drops_buildings_that_do_not_classify() -> None:
    buildings = _buildings(
        [
            {
                "osm_id": "1",
                "osm_type": "way",
                "lat": 55.78,
                "lon": 49.12,
                "building": "yes",
                "levels": None,
                "flats": None,
            },
            {
                "osm_id": "2",
                "osm_type": "way",
                "lat": 55.79,
                "lon": 49.13,
                "building": "garage",
                "levels": None,
                "flats": None,
            },
            {
                "osm_id": "3",
                "osm_type": "way",
                "lat": 55.80,
                "lon": 49.14,
                "building": "apartments",
                "levels": "5",
                "flats": "20",
            },
        ]
    )

    result = assemble_valuation_objects(buildings)

    assert result.height == 1
    assert result["object_id"][0] == "way/3"


def test_object_id_is_osm_type_slash_id() -> None:
    buildings = _buildings(
        [
            {
                "osm_id": "42",
                "osm_type": "relation",
                "lat": 55.78,
                "lon": 49.12,
                "building": "house",
                "levels": "1",
                "flats": None,
            }
        ]
    )

    result = assemble_valuation_objects(buildings)

    assert result["object_id"][0] == "relation/42"


def test_asset_class_uses_enum_value() -> None:
    buildings = _buildings(
        [
            {
                "osm_id": "1",
                "osm_type": "way",
                "lat": 55.78,
                "lon": 49.12,
                "building": "apartments",
                "levels": None,
                "flats": None,
            },
            {
                "osm_id": "2",
                "osm_type": "way",
                "lat": 55.78,
                "lon": 49.12,
                "building": "detached",
                "levels": None,
                "flats": None,
            },
            {
                "osm_id": "3",
                "osm_type": "way",
                "lat": 55.78,
                "lon": 49.12,
                "building": "retail",
                "levels": None,
                "flats": None,
            },
        ]
    )

    result = assemble_valuation_objects(buildings)

    assert sorted(result["asset_class"].to_list()) == sorted(
        [AssetClass.APARTMENT.value, AssetClass.HOUSE.value, AssetClass.COMMERCIAL.value]
    )


def test_levels_and_flats_cast_to_int_with_nulls() -> None:
    buildings = _buildings(
        [
            {
                "osm_id": "1",
                "osm_type": "way",
                "lat": 55.78,
                "lon": 49.12,
                "building": "apartments",
                "levels": "9",
                "flats": "72",
            },
            {
                "osm_id": "2",
                "osm_type": "way",
                "lat": 55.79,
                "lon": 49.13,
                "building": "apartments",
                "levels": None,
                "flats": None,
            },
        ]
    )

    result = assemble_valuation_objects(buildings)

    assert result["levels"].dtype == pl.Int64
    assert result["flats"].dtype == pl.Int64
    levels_sorted = sorted(
        result["levels"].to_list(), key=lambda v: (v is None, v)
    )
    assert levels_sorted == [9, None]


def test_drops_rows_with_invalid_lat_lon() -> None:
    buildings = _buildings(
        [
            {
                "osm_id": "1",
                "osm_type": "way",
                "lat": 95.0,  # invalid
                "lon": 49.12,
                "building": "apartments",
                "levels": None,
                "flats": None,
            },
            {
                "osm_id": "2",
                "osm_type": "way",
                "lat": 55.78,
                "lon": 49.12,
                "building": "apartments",
                "levels": None,
                "flats": None,
            },
        ]
    )

    result = assemble_valuation_objects(buildings)

    assert result.height == 1
    assert result["object_id"][0] == "way/2"


def test_empty_input_returns_empty_with_schema() -> None:
    result = assemble_valuation_objects(_buildings([]))

    assert result.is_empty()
    assert set(result.columns) == {
        "object_id",
        "asset_class",
        "lat",
        "lon",
        "levels",
        "flats",
    }
