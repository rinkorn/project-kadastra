"""Tests for compute_object_polygon_features.

For each (polygon layer, radius) pair the function attaches a column
``{layer}_share_{R}m`` whose value is the share of the buffer-circle
area at the object covered by polygons from that layer. ADR-0014
explains the design (UTM 39N projection, STRtree, 4 layers × 4 radii).

Tests use small fixtures around Kazan and rely on UTM-zone-39N
projection for area calculation; the helper internally projects.
"""

from __future__ import annotations

import polars as pl
import pytest
from pyproj import Transformer
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from kadastra.etl.object_polygon_features import compute_object_polygon_features

KAZAN_LAT, KAZAN_LON = 55.7887, 49.1221

# WGS84 → UTM 39N. Used in tests to construct WGS84 polygons of known
# area (and shape) by projecting from a metric frame.
_TO_UTM = Transformer.from_crs("EPSG:4326", "EPSG:32639", always_xy=True)
_FROM_UTM = Transformer.from_crs("EPSG:32639", "EPSG:4326", always_xy=True)


def _objects(rows: list[dict[str, object]]) -> pl.DataFrame:
    schema = {
        "object_id": pl.Utf8,
        "lat": pl.Float64,
        "lon": pl.Float64,
    }
    return pl.DataFrame(rows, schema=schema)


def _utm_square_around(base_lat: float, base_lon: float, *, side_m: float, offset_east_m: float = 0.0) -> Polygon:
    """Build a WGS84 polygon that is a metric-square in UTM 39N around the
    base point, optionally offset to the east. Useful for tests that need
    an exact area."""
    cx, cy = _TO_UTM.transform(base_lon, base_lat)
    cx += offset_east_m
    half = side_m / 2.0
    corners_utm = [
        (cx - half, cy - half),
        (cx + half, cy - half),
        (cx + half, cy + half),
        (cx - half, cy + half),
    ]
    corners_wgs = [_FROM_UTM.transform(x, y) for x, y in corners_utm]
    return Polygon([(lon, lat) for lon, lat in corners_wgs])


def test_adds_share_columns_per_layer_and_radius() -> None:
    objects = _objects([{"object_id": "a", "lat": KAZAN_LAT, "lon": KAZAN_LON}])
    big_water = _utm_square_around(KAZAN_LAT, KAZAN_LON, side_m=4_000.0)

    out = compute_object_polygon_features(
        objects,
        polygons_by_layer={"water": [big_water]},
        radii_m=[100, 300, 500, 800],
    )

    expected = {
        "water_share_100m",
        "water_share_300m",
        "water_share_500m",
        "water_share_800m",
    }
    assert expected.issubset(set(out.columns))


def test_full_coverage_yields_share_one() -> None:
    """Object sits in the middle of a 4 km square — its 800 m buffer is
    fully inside the polygon, so every radius gets share == 1.0."""
    objects = _objects([{"object_id": "a", "lat": KAZAN_LAT, "lon": KAZAN_LON}])
    big = _utm_square_around(KAZAN_LAT, KAZAN_LON, side_m=4_000.0)

    out = compute_object_polygon_features(
        objects,
        polygons_by_layer={"poly": [big]},
        radii_m=[100, 300, 800],
    )

    assert out["poly_share_100m"][0] == pytest.approx(1.0, abs=0.01)
    assert out["poly_share_300m"][0] == pytest.approx(1.0, abs=0.01)
    assert out["poly_share_800m"][0] == pytest.approx(1.0, abs=0.01)


def test_no_intersection_yields_share_zero() -> None:
    """Polygon sits 5 km away — no overlap with any of the buffers."""
    objects = _objects([{"object_id": "a", "lat": KAZAN_LAT, "lon": KAZAN_LON}])
    far_poly = _utm_square_around(KAZAN_LAT, KAZAN_LON, side_m=200.0, offset_east_m=5_000.0)

    out = compute_object_polygon_features(
        objects,
        polygons_by_layer={"poly": [far_poly]},
        radii_m=[100, 800],
    )

    assert out["poly_share_100m"][0] == 0.0
    assert out["poly_share_800m"][0] == 0.0


def test_partial_coverage_grows_then_shrinks_with_radius() -> None:
    """A 200 m square 200 m east of the object covers part of the 300/500/800
    buffers but none of the 100 (too far)."""
    objects = _objects([{"object_id": "a", "lat": KAZAN_LAT, "lon": KAZAN_LON}])
    poly = _utm_square_around(KAZAN_LAT, KAZAN_LON, side_m=200.0, offset_east_m=200.0)

    out = compute_object_polygon_features(
        objects,
        polygons_by_layer={"poly": [poly]},
        radii_m=[100, 300, 500, 800],
    )

    # 100 m buffer reaches 100 m east; the polygon starts 100 m east. They
    # touch only at a tangent — share should be ~0.
    assert out["poly_share_100m"][0] == pytest.approx(0.0, abs=0.01)
    # 300 m / 500 m / 800 m all overlap but with smaller share as buffer grows
    # (polygon area is fixed at 200x200 = 40000 sq.m; buffer area grows ~R^2).
    assert out["poly_share_300m"][0] > 0.0
    assert out["poly_share_300m"][0] > out["poly_share_500m"][0]
    assert out["poly_share_500m"][0] > out["poly_share_800m"][0]


def test_disjoint_polygons_of_same_layer_sum() -> None:
    """Two non-overlapping polygons of the same layer contribute additively
    to the share. (This is the raison d'être of poly-area: cumulative
    'water on both sides' counts as more than 'water on one side'.)"""
    objects = _objects([{"object_id": "a", "lat": KAZAN_LAT, "lon": KAZAN_LON}])
    poly_east = _utm_square_around(KAZAN_LAT, KAZAN_LON, side_m=200.0, offset_east_m=200.0)
    poly_west = _utm_square_around(KAZAN_LAT, KAZAN_LON, side_m=200.0, offset_east_m=-200.0)

    out = compute_object_polygon_features(
        objects,
        polygons_by_layer={"water": [poly_east, poly_west]},
        radii_m=[800],
    )
    out_single = compute_object_polygon_features(
        objects,
        polygons_by_layer={"water": [poly_east]},
        radii_m=[800],
    )

    assert out["water_share_800m"][0] == pytest.approx(2.0 * out_single["water_share_800m"][0], rel=0.05)


def test_layers_are_independent() -> None:
    """A polygon in one layer must not affect another layer's share."""
    objects = _objects([{"object_id": "a", "lat": KAZAN_LAT, "lon": KAZAN_LON}])
    big = _utm_square_around(KAZAN_LAT, KAZAN_LON, side_m=4_000.0)

    out = compute_object_polygon_features(
        objects,
        polygons_by_layer={"water": [big], "park": []},
        radii_m=[300],
    )

    assert out["water_share_300m"][0] == pytest.approx(1.0, abs=0.01)
    assert out["park_share_300m"][0] == 0.0


def test_empty_objects_returns_empty_with_columns() -> None:
    objects = _objects([])
    layer: list[BaseGeometry] = [_utm_square_around(KAZAN_LAT, KAZAN_LON, side_m=200.0)]

    out = compute_object_polygon_features(objects, polygons_by_layer={"water": layer}, radii_m=[100, 800])

    assert out.is_empty()
    assert "water_share_100m" in out.columns
    assert "water_share_800m" in out.columns


def test_no_layers_returns_input_unchanged() -> None:
    objects = _objects([{"object_id": "a", "lat": KAZAN_LAT, "lon": KAZAN_LON}])

    out = compute_object_polygon_features(objects, polygons_by_layer={}, radii_m=[100, 800])

    assert out.columns == objects.columns
    assert out.height == objects.height


def test_share_is_in_zero_one_range_per_layer() -> None:
    """Even with a huge polygon spanning many km, the share for a single
    layer must not exceed 1.0 — a layer's polygons are merged for the
    intersection-area numerator (or we just clip)."""
    objects = _objects([{"object_id": "a", "lat": KAZAN_LAT, "lon": KAZAN_LON}])
    # Two overlapping huge squares — naive sum would give > 1.
    a = _utm_square_around(KAZAN_LAT, KAZAN_LON, side_m=4_000.0)
    b = _utm_square_around(KAZAN_LAT, KAZAN_LON, side_m=4_000.0, offset_east_m=200.0)

    out = compute_object_polygon_features(
        objects,
        polygons_by_layer={"poly": [a, b]},
        radii_m=[300, 800],
    )

    assert 0.0 <= out["poly_share_300m"][0] <= 1.0 + 1e-9
    assert 0.0 <= out["poly_share_800m"][0] <= 1.0 + 1e-9


def test_full_coverage_share_is_exactly_one() -> None:
    """Sanity: when the layer fully contains the buffer, share == 1.0
    exactly (modulo floating-point), independent of the polygon
    discretization shapely uses to approximate the circle. The
    implementation normalizes by the buffer's actual area, not by
    π·r², so this holds at any quad_segs value."""
    objects = _objects([{"object_id": "a", "lat": KAZAN_LAT, "lon": KAZAN_LON}])
    R = 300
    # Big enough to fully cover the 300 m buffer.
    poly = _utm_square_around(KAZAN_LAT, KAZAN_LON, side_m=2 * R + 200)

    out = compute_object_polygon_features(objects, polygons_by_layer={"poly": [poly]}, radii_m=[R])

    assert out[f"poly_share_{R}m"][0] == pytest.approx(1.0, abs=1e-6)
