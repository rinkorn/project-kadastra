"""Unit tests for ADR-0019 — distance from each object to the nearest
geometry of every layer (Polygon, LineString, or Point).

Complements the existing ``compute_object_polygon_features`` (share)
with an absolute-distance signal: «5 м до парка» is qualitatively
different from «100 м до парка» even when the share-in-buffer is
identical (object covered by the same fraction of greenery within
500 m).

Distances are in metres in EPSG:32639 (UTM-39N); same projection as
the share computation. The helper is geometry-agnostic so that point
POIs (school/kindergarten/...), linear externalities (powerline,
railway) and polygonal layers (water/park/landfill) can share one
pipeline."""

from __future__ import annotations

import polars as pl
from shapely.geometry import LineString, Point, Polygon, box
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


def test_distance_to_point_layer() -> None:
    """Point POIs (e.g. school, kindergarten) — distance to nearest
    OSM node. Two points 0.005° east and 0.020° west of the object;
    the helper must pick the east (closer) one (~315 m at 55.79°)."""
    near = Point(_KAZAN_LON + 0.005, _KAZAN_LAT)
    far = Point(_KAZAN_LON - 0.020, _KAZAN_LAT)
    df = compute_object_geom_distance_features(
        _objects([(_KAZAN_LAT, _KAZAN_LON)]),
        geometries_by_layer={"school": [far, near]},
    )
    d = float(df["dist_to_school_m"][0])
    # 0.005° lon × cos(55.79°) × 111 km/° ≈ 314 m. ±5 % tolerance for
    # the spheroid mismatch with the simple cosine approximation.
    assert 295.0 < d < 335.0


def test_distance_to_linestring_layer() -> None:
    """Linear features (powerline, railway). Object 0.005° south of a
    long west-east line at _KAZAN_LAT; expected ~555 m (lat-only delta,
    constant across longitudes)."""
    line = LineString(
        [(_KAZAN_LON - 0.05, _KAZAN_LAT), (_KAZAN_LON + 0.05, _KAZAN_LAT)]
    )
    df = compute_object_geom_distance_features(
        _objects([(_KAZAN_LAT - 0.005, _KAZAN_LON)]),
        geometries_by_layer={"railway": [line]},
    )
    d = float(df["dist_to_railway_m"][0])
    # 0.005° lat × 111 km/° ≈ 555 m. Latitude axis is independent of
    # longitude scale, so the tolerance can be tighter.
    assert 545.0 < d < 565.0


def test_object_on_linestring_distance_is_zero() -> None:
    """If the object is exactly on the line, distance is 0."""
    line = LineString(
        [(_KAZAN_LON - 0.05, _KAZAN_LAT), (_KAZAN_LON + 0.05, _KAZAN_LAT)]
    )
    df = compute_object_geom_distance_features(
        _objects([(_KAZAN_LAT, _KAZAN_LON)]),
        geometries_by_layer={"railway": [line]},
    )
    assert float(df["dist_to_railway_m"][0]) < 1.0


def test_mixed_geometry_types_in_one_call() -> None:
    """Single call mixing point, line and polygon layers — exactly how
    the production usecase wires school/railway/water side by side."""
    school = Point(_KAZAN_LON + 0.005, _KAZAN_LAT)
    railway = LineString(
        [(_KAZAN_LON - 0.05, _KAZAN_LAT - 0.005),
         (_KAZAN_LON + 0.05, _KAZAN_LAT - 0.005)]
    )
    water = _polygon_around(_KAZAN_LAT + 0.010, _KAZAN_LON, 0.001)
    df = compute_object_geom_distance_features(
        _objects([(_KAZAN_LAT, _KAZAN_LON)]),
        geometries_by_layer={
            "school": [school],
            "railway": [railway],
            "water": [water],
        },
    )
    assert df["dist_to_school_m"][0] is not None
    assert df["dist_to_railway_m"][0] is not None
    assert df["dist_to_water_m"][0] is not None
    # All three are positive and ordered roughly by their geometric
    # construction: school 0.005° east ≈ 315 m, railway 0.005° south
    # ≈ 555 m, water 0.010° north ≈ 1100 m.
    assert float(df["dist_to_school_m"][0]) < float(df["dist_to_railway_m"][0])
    assert float(df["dist_to_railway_m"][0]) < float(df["dist_to_water_m"][0])
