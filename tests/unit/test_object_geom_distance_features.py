"""Unit tests for ADR-0019 — distance from each object to the nearest
polygon of every layer.

Complements the existing ``compute_object_polygon_features`` (share)
with an absolute-distance signal: «5 м до парка» is qualitatively
different from «100 м до парка» even when the share-in-buffer is
identical (object covered by the same fraction of greenery within
500 m).

Distances are in metres in EPSG:32639 (UTM-39N); same projection as
the share computation."""

from __future__ import annotations

import polars as pl
from shapely.geometry import Polygon, box
from shapely.geometry.base import BaseGeometry

from kadastra.etl.object_geom_distance_features import (
    compute_object_geom_distance_features,
)

# Kazan reference; objects are placed off it by tiny lat/lon deltas
# whose UTM-39N metric distance is well-defined and predictable.
_KAZAN_LAT = 55.7905
_KAZAN_LON = 49.1142


def _objects(coords: list[tuple[float, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "object_id": [f"way/{i}" for i in range(len(coords))],
            "lat": [lat for lat, _ in coords],
            "lon": [lon for _, lon in coords],
        }
    )


def _polygon_around(lat: float, lon: float, radius_deg: float) -> Polygon:
    """Approximate square polygon around a point in WGS84 degrees.
    Used for tests where exact geometry is not the assertion target —
    we just need a polygon that contains/excludes the test points."""
    return box(
        lon - radius_deg, lat - radius_deg, lon + radius_deg, lat + radius_deg
    )


def test_empty_layer_yields_null_column() -> None:
    df = compute_object_geom_distance_features(
        _objects([(_KAZAN_LAT, _KAZAN_LON)]),
        geometries_by_layer={"water": []},
    )
    assert "dist_to_water_m" in df.columns
    assert df["dist_to_water_m"][0] is None


def test_object_inside_polygon_distance_is_zero() -> None:
    poly = _polygon_around(_KAZAN_LAT, _KAZAN_LON, 0.001)  # ~110 m square
    df = compute_object_geom_distance_features(
        _objects([(_KAZAN_LAT, _KAZAN_LON)]),
        geometries_by_layer={"water": [poly]},
    )
    assert df["dist_to_water_m"][0] == 0.0


def test_object_outside_polygon_distance_is_positive() -> None:
    """Object 0.01° east of polygon centre; polygon 0.001° around centre.
    The east edge of the polygon is at lon + 0.001°; the object is at
    lon + 0.01°. Δ_lon = 0.009°. At φ ≈ 55.79° this is roughly 567 m."""
    poly = _polygon_around(_KAZAN_LAT, _KAZAN_LON, 0.001)
    df = compute_object_geom_distance_features(
        _objects([(_KAZAN_LAT, _KAZAN_LON + 0.01)]),
        geometries_by_layer={"water": [poly]},
    )
    d = float(df["dist_to_water_m"][0])
    # ~567 m at 55.79° latitude — give a 5% tolerance for the
    # mercator/UTM/spheroid mismatch in the approximation above.
    assert 540.0 < d < 595.0


def test_distance_to_nearest_of_multiple_polygons() -> None:
    """Two non-overlapping polygons 0.005° east and 0.020° west of
    object. Object should report distance to the closer (east) one."""
    near = _polygon_around(_KAZAN_LAT, _KAZAN_LON + 0.005, 0.001)
    far = _polygon_around(_KAZAN_LAT, _KAZAN_LON - 0.020, 0.001)
    df = compute_object_geom_distance_features(
        _objects([(_KAZAN_LAT, _KAZAN_LON)]),
        geometries_by_layer={"park": [far, near]},
    )
    d = float(df["dist_to_park_m"][0])
    # Near polygon: east edge at lon + 0.004°, object at lon + 0° →
    # ~250 m. Far polygon: east edge at lon − 0.019°, object at 0 →
    # ~1190 m. We expect the near one.
    assert 230.0 < d < 275.0


def test_multiple_layers_yield_one_column_each() -> None:
    poly_water = _polygon_around(_KAZAN_LAT, _KAZAN_LON + 0.010, 0.001)
    poly_park = _polygon_around(_KAZAN_LAT + 0.010, _KAZAN_LON, 0.001)
    df = compute_object_geom_distance_features(
        _objects([(_KAZAN_LAT, _KAZAN_LON)]),
        geometries_by_layer={"water": [poly_water], "park": [poly_park]},
    )
    assert "dist_to_water_m" in df.columns
    assert "dist_to_park_m" in df.columns
    assert df["dist_to_water_m"][0] is not None
    assert df["dist_to_park_m"][0] is not None


def test_preserves_existing_columns() -> None:
    df_in = pl.DataFrame(
        {
            "object_id": ["w/1", "w/2"],
            "lat": [_KAZAN_LAT, _KAZAN_LAT + 0.001],
            "lon": [_KAZAN_LON, _KAZAN_LON],
            "asset_class": ["apartment", "house"],
        }
    )
    poly = _polygon_around(_KAZAN_LAT, _KAZAN_LON, 0.001)
    df_out = compute_object_geom_distance_features(
        df_in,
        geometries_by_layer={"water": [poly]},
    )
    assert df_out["object_id"].to_list() == ["w/1", "w/2"]
    assert df_out["asset_class"].to_list() == ["apartment", "house"]


def test_no_layers_returns_unchanged_frame() -> None:
    df_in = _objects([(_KAZAN_LAT, _KAZAN_LON)])
    df_out = compute_object_geom_distance_features(
        df_in, geometries_by_layer={}
    )
    assert df_out.columns == df_in.columns


def test_empty_objects_frame_emits_null_columns() -> None:
    """When objects is empty, columns must still appear (with no rows)
    so downstream schema is stable per asset_class slice."""
    empty: dict[str, list[BaseGeometry]] = {
        "water": [_polygon_around(_KAZAN_LAT, _KAZAN_LON, 0.001)],
        "landfill": [],
    }
    df_in = pl.DataFrame(
        {"object_id": [], "lat": [], "lon": []},
        schema={
            "object_id": pl.Utf8,
            "lat": pl.Float64,
            "lon": pl.Float64,
        },
    )
    df_out = compute_object_geom_distance_features(
        df_in, geometries_by_layer=empty
    )
    assert "dist_to_water_m" in df_out.columns
    assert "dist_to_landfill_m" in df_out.columns
    assert df_out.height == 0
