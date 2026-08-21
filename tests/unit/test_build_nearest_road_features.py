"""Tests for compute_nearest_road_features (ADR-0024, group 1).

Synthetic ways: the expected nearest_road_class / dist_to_* values are
derived from hand-placed geometries around Kazan. Expected distances
use haversine (radial distances) with a tolerance that covers the
haversine↔UTM-39N discrepancy at this latitude.
"""

import polars as pl
import pytest

from kadastra.etl.haversine import haversine_meters
from kadastra.etl.object_road_class_features import (
    DIST_CLASS_GROUPS,
    compute_nearest_road_features,
    normalize_highway_class,
)

KAZAN_LAT, KAZAN_LON = 55.7887, 49.1221

# Residential street: short N-S segment the test objects sit next to.
RESIDENTIAL_WAY = [(KAZAN_LAT, KAZAN_LON), (KAZAN_LAT + 0.001, KAZAN_LON)]
# Motorway: E-W segment ~1.1 km north of the objects.
MOTORWAY_WAY = [
    (KAZAN_LAT + 0.01, KAZAN_LON - 0.01),
    (KAZAN_LAT + 0.01, KAZAN_LON + 0.01),
]
# Footway: E-W segment ~110 m south — pedestrian infrastructure union.
FOOTWAY_WAY = [
    (KAZAN_LAT - 0.001, KAZAN_LON - 0.005),
    (KAZAN_LAT - 0.001, KAZAN_LON + 0.005),
]


def _objects(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={"object_id": pl.Utf8, "lat": pl.Float64, "lon": pl.Float64},
    )


def test_nearest_class_and_distances() -> None:
    objects = _objects(
        [
            # Sits exactly on the residential way.
            {"object_id": "a", "lat": KAZAN_LAT + 0.0005, "lon": KAZAN_LON},
        ]
    )
    ways = {
        "residential": [RESIDENTIAL_WAY],
        "motorway": [MOTORWAY_WAY],
        "pedestrian": [FOOTWAY_WAY],
    }

    out = compute_nearest_road_features(objects, ways_by_class=ways)

    assert out.columns == [
        "object_id",
        "nearest_road_class",
        *DIST_CLASS_GROUPS.keys(),
    ]
    row = out.row(0, named=True)
    assert row["nearest_road_class"] == "residential"
    assert row["dist_to_residential_m"] == pytest.approx(0.0, abs=1.0)
    expected_motorway = haversine_meters(KAZAN_LAT + 0.0005, KAZAN_LON, KAZAN_LAT + 0.01, KAZAN_LON)
    assert row["dist_to_motorway_m"] == pytest.approx(expected_motorway, rel=0.05)
    expected_footway = haversine_meters(KAZAN_LAT + 0.0005, KAZAN_LON, KAZAN_LAT - 0.001, KAZAN_LON)
    assert row["dist_to_pedestrian_m"] == pytest.approx(expected_footway, rel=0.05)
    # No primary/secondary ways → null distances.
    assert row["dist_to_primary_m"] is None
    assert row["dist_to_secondary_m"] is None


def test_nearest_class_picks_closest_way_across_classes() -> None:
    # Object ~20 m from the motorway line, ~1 km from residential.
    objects = _objects(
        [
            {"object_id": "b", "lat": KAZAN_LAT + 0.0098, "lon": KAZAN_LON},
        ]
    )
    ways = {
        "residential": [RESIDENTIAL_WAY],
        "motorway": [MOTORWAY_WAY],
    }

    out = compute_nearest_road_features(objects, ways_by_class=ways)

    row = out.row(0, named=True)
    assert row["nearest_road_class"] == "motorway"
    assert row["dist_to_motorway_m"] == pytest.approx(
        haversine_meters(KAZAN_LAT + 0.0098, KAZAN_LON, KAZAN_LAT + 0.01, KAZAN_LON),
        rel=0.05,
    )


def test_motorway_group_includes_trunk() -> None:
    objects = _objects([{"object_id": "c", "lat": KAZAN_LAT, "lon": KAZAN_LON}])
    ways = {"trunk": [MOTORWAY_WAY]}

    out = compute_nearest_road_features(objects, ways_by_class=ways)

    row = out.row(0, named=True)
    assert row["nearest_road_class"] == "trunk"
    assert row["dist_to_motorway_m"] == pytest.approx(
        haversine_meters(KAZAN_LAT, KAZAN_LON, KAZAN_LAT + 0.01, KAZAN_LON),
        rel=0.05,
    )


def test_empty_ways_produce_null_columns() -> None:
    objects = _objects([{"object_id": "d", "lat": KAZAN_LAT, "lon": KAZAN_LON}])

    out = compute_nearest_road_features(objects, ways_by_class={})

    assert out.height == 1
    row = out.row(0, named=True)
    assert row["nearest_road_class"] is None
    for col in DIST_CLASS_GROUPS:
        assert row[col] is None


def test_null_coordinates_get_null_features() -> None:
    objects = _objects(
        [
            {"object_id": "e", "lat": None, "lon": None},
            {"object_id": "f", "lat": KAZAN_LAT, "lon": KAZAN_LON},
        ]
    )
    ways = {"residential": [RESIDENTIAL_WAY]}

    out = compute_nearest_road_features(objects, ways_by_class=ways)

    null_row = out.filter(pl.col("object_id") == "e").row(0, named=True)
    assert null_row["nearest_road_class"] is None
    assert null_row["dist_to_residential_m"] is None
    ok_row = out.filter(pl.col("object_id") == "f").row(0, named=True)
    assert ok_row["nearest_road_class"] == "residential"


def test_empty_objects_frame() -> None:
    objects = _objects([])
    out = compute_nearest_road_features(objects, ways_by_class={"residential": [RESIDENTIAL_WAY]})
    assert out.height == 0
    assert out.columns == ["object_id", "nearest_road_class", *DIST_CLASS_GROUPS.keys()]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("motorway", "motorway"),
        ("motorway_link", "motorway"),
        ("trunk_link", "trunk"),
        ("primary_link", "primary"),
        ("secondary_link", "secondary"),
        ("tertiary_link", "tertiary"),
        ("residential", "residential"),
        ("service", "service"),
        ("unclassified", "unclassified"),
        ("pedestrian", "pedestrian"),
        ("footway", "pedestrian"),
        ("living_street", "pedestrian"),
        ("cycleway", None),
        ("track", None),
        ("construction", None),
        (None, None),
    ],
)
def test_normalize_highway_class(raw: str | None, expected: str | None) -> None:
    assert normalize_highway_class(raw) == expected
