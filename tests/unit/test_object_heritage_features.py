"""Unit tests for the heritage (ОКН) feature block (ADR-0025, п. 2).

Fixtures use synthetic ОКН at real Kazan coordinates: ~22 m and ~490 m
north of the test object, plus one far away — so dist/count/flag
expectations are readable in metres (1° lat ≈ 111 320 m).
"""

from __future__ import annotations

import json

import polars as pl
import pytest

from kadastra.etl.object_heritage_features import (
    HERITAGE_FEATURE_COLUMNS,
    HERITAGE_SILVER_SCHEMA,
    compute_object_heritage_features,
    parse_heritage_geojsonseq,
)

# Test object anchor (Казань, центр).
OBJ_LAT, OBJ_LON = 55.7900, 49.1000
# ~22 m north of the object.
NEAR_LAT = 55.7902
# ~490 m north of the object — inside the 500 m count radius.
MID_LAT = 55.7944
# Far away (Новое Караваево side).
FAR_LAT, FAR_LON = 55.8500, 49.2000


def _objects(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={"object_id": pl.Utf8, "lat": pl.Float64, "lon": pl.Float64},
    )


def _heritage(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=HERITAGE_SILVER_SCHEMA)


def _point_row(osm_id: str, lat: float, lon: float) -> dict[str, object]:
    return {
        "osm_id": osm_id,
        "ref_egrokn": None,
        "heritage_level": "2",
        "name": None,
        "lat": lat,
        "lon": lon,
        "polygon_wkt": None,
    }


# ---------------------------------------------------------------- parse


def test_parse_filters_non_heritage_rows() -> None:
    lines = [
        json.dumps(
            {
                "type": "Feature",
                "id": "n1",
                "geometry": {"type": "Point", "coordinates": [49.1, 55.79]},
                "properties": {"heritage": "2", "ref:egrokn": "161510400850006", "name": "Памятник"},
            }
        ),
        json.dumps(
            {
                "type": "Feature",
                "id": "n2",
                "geometry": {"type": "Point", "coordinates": [49.1, 55.79]},
                "properties": {"barrier": "gate"},
            }
        ),
    ]
    df = parse_heritage_geojsonseq(lines)
    assert df.height == 1
    row = df.row(0, named=True)
    assert row["osm_id"] == "n1"
    assert row["ref_egrokn"] == "161510400850006"
    assert row["heritage_level"] == "2"
    assert row["name"] == "Памятник"
    assert row["lat"] == pytest.approx(55.79)
    assert row["lon"] == pytest.approx(49.1)
    assert row["polygon_wkt"] is None


def test_parse_promotes_closed_linestring_to_polygon() -> None:
    ring = [[49.1, 55.79], [49.101, 55.79], [49.101, 55.7905], [49.1, 55.7905], [49.1, 55.79]]
    lines = [
        json.dumps(
            {
                "type": "Feature",
                "id": "w1",
                "geometry": {"type": "LineString", "coordinates": ring},
                "properties": {"heritage": "6", "building": "apartments"},
            }
        )
    ]
    df = parse_heritage_geojsonseq(lines)
    assert df.height == 1
    wkt = df["polygon_wkt"][0]
    assert wkt is not None and wkt.startswith("POLYGON")


def test_parse_multipolygon_gets_wkt_and_centroid() -> None:
    mp = [[[[49.1, 55.79], [49.102, 55.79], [49.102, 55.791], [49.1, 55.791], [49.1, 55.79]]]]
    lines = [
        json.dumps(
            {
                "type": "Feature",
                "id": "r1",
                "geometry": {"type": "MultiPolygon", "coordinates": mp},
                "properties": {"heritage": "1"},
            }
        )
    ]
    df = parse_heritage_geojsonseq(lines)
    row = df.row(0, named=True)
    assert row["polygon_wkt"] is not None and "MULTIPOLYGON" in row["polygon_wkt"]
    assert row["lat"] == pytest.approx(55.7905, abs=1e-3)
    assert row["lon"] == pytest.approx(49.101, abs=1e-3)


def test_parse_empty_input_gives_schema_frame() -> None:
    df = parse_heritage_geojsonseq([])
    assert df.height == 0
    assert df.schema == HERITAGE_SILVER_SCHEMA


def test_parse_handles_rs_prefix_and_blank_lines() -> None:
    feature = json.dumps(
        {
            "type": "Feature",
            "id": "n1",
            "geometry": {"type": "Point", "coordinates": [49.1, 55.79]},
            "properties": {"heritage": "2"},
        }
    )
    df = parse_heritage_geojsonseq(["\x1e" + feature, "", "  "])
    assert df.height == 1


# -------------------------------------------------------------- compute


def test_nearest_dist_count_and_object_flag() -> None:
    heritage = _heritage(
        [
            _point_row("n1", NEAR_LAT, OBJ_LON),
            _point_row("n2", MID_LAT, OBJ_LON),
            _point_row("n3", FAR_LAT, FAR_LON),
        ]
    )
    objects = _objects([{"object_id": "a", "lat": OBJ_LAT, "lon": OBJ_LON}])
    df = compute_object_heritage_features(objects, heritage=heritage)
    row = df.row(0, named=True)
    # Nearest is ~22 m away (within the 50 m object buffer).
    assert 15.0 < row["dist_to_nearest_heritage_m"] < 30.0
    assert row["is_heritage_object"] == 1
    # Near + mid are within 500 m, far is not.
    assert row["count_heritage_500m"] == 2


def test_distances_are_utm_metres_not_degrees() -> None:
    heritage = _heritage([_point_row("n1", MID_LAT, OBJ_LON)])
    objects = _objects([{"object_id": "a", "lat": OBJ_LAT, "lon": OBJ_LON}])
    df = compute_object_heritage_features(objects, heritage=heritage)
    assert df["dist_to_nearest_heritage_m"][0] == pytest.approx(489.0, abs=10.0)


def test_inside_heritage_zone_polygon_mode() -> None:
    """With polygonal ОКН in the layer, the flag is pure containment —
    no distance fallback mixing (ADR-0025 «Открытые вопросы»)."""
    # Square footprint around the object: ±0.0003 lat, ±0.0005 lon.
    ring = (
        f"{OBJ_LON - 0.0005} {OBJ_LAT - 0.0003}, "
        f"{OBJ_LON + 0.0005} {OBJ_LAT - 0.0003}, "
        f"{OBJ_LON + 0.0005} {OBJ_LAT + 0.0003}, "
        f"{OBJ_LON - 0.0005} {OBJ_LAT + 0.0003}, "
        f"{OBJ_LON - 0.0005} {OBJ_LAT - 0.0003}"
    )
    heritage = _heritage(
        [
            {
                "osm_id": "r1",
                "ref_egrokn": None,
                "heritage_level": "1",
                "name": None,
                "lat": OBJ_LAT,
                "lon": OBJ_LON,
                "polygon_wkt": f"MULTIPOLYGON ((({ring})))",
            }
        ]
    )
    objects = _objects(
        [
            {"object_id": "inside", "lat": OBJ_LAT, "lon": OBJ_LON},
            {"object_id": "outside", "lat": FAR_LAT, "lon": FAR_LON},
        ]
    )
    df = compute_object_heritage_features(objects, heritage=heritage)
    rows = {r["object_id"]: r for r in df.iter_rows(named=True)}
    assert rows["inside"]["inside_heritage_zone"] == 1
    assert rows["outside"]["inside_heritage_zone"] == 0


def test_inside_heritage_zone_distance_fallback() -> None:
    """Point-only layer (no polygons) → fallback dist < 100 m."""
    heritage = _heritage([_point_row("n1", NEAR_LAT, OBJ_LON)])
    objects = _objects(
        [
            {"object_id": "near", "lat": OBJ_LAT, "lon": OBJ_LON},
            {"object_id": "mid", "lat": MID_LAT, "lon": OBJ_LON},
        ]
    )
    df = compute_object_heritage_features(objects, heritage=heritage)
    rows = {r["object_id"]: r for r in df.iter_rows(named=True)}
    assert rows["near"]["inside_heritage_zone"] == 1  # ~22 m
    assert rows["mid"]["inside_heritage_zone"] == 0  # ~468 m from the point


def test_empty_heritage_layer_gives_null_features() -> None:
    objects = _objects([{"object_id": "a", "lat": OBJ_LAT, "lon": OBJ_LON}])
    df = compute_object_heritage_features(objects, heritage=_heritage([]))
    for col in HERITAGE_FEATURE_COLUMNS:
        assert col in df.columns
        assert df[col][0] is None


def test_null_coords_get_null_features() -> None:
    heritage = _heritage([_point_row("n1", NEAR_LAT, OBJ_LON)])
    objects = _objects([{"object_id": "a", "lat": None, "lon": None}])
    df = compute_object_heritage_features(objects, heritage=heritage)
    for col in HERITAGE_FEATURE_COLUMNS:
        assert df[col][0] is None


def test_feature_column_dtypes() -> None:
    heritage = _heritage([_point_row("n1", NEAR_LAT, OBJ_LON)])
    objects = _objects([{"object_id": "a", "lat": OBJ_LAT, "lon": OBJ_LON}])
    df = compute_object_heritage_features(objects, heritage=heritage)
    assert df.schema["is_heritage_object"] == pl.Int64
    assert df.schema["inside_heritage_zone"] == pl.Int64
    assert df.schema["count_heritage_500m"] == pl.Int64
    assert df.schema["dist_to_nearest_heritage_m"] == pl.Float64
