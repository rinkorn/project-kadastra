"""Tests for compute_object_zonal_features.

For each (layer, radius) pair the function adds a column named
``{layer}_within_{R}m`` whose value is the number of points from
``layers[layer]`` within ``R`` meters (haversine) of the object.

Self-exclusion: if a layer DataFrame contains an ``object_id`` column,
rows whose ``object_id`` matches the current object's are not counted.
This is how per-class density (apartments→apartments, etc.) avoids
double-counting the object itself.

ADR-0013 has the design rationale.
"""

from __future__ import annotations

import polars as pl

from kadastra.etl.haversine import haversine_meters
from kadastra.etl.object_zonal_features import compute_object_zonal_features

KAZAN_LAT, KAZAN_LON = 55.7887, 49.1221


def _objects(rows: list[dict[str, object]]) -> pl.DataFrame:
    schema = {
        "object_id": pl.Utf8,
        "lat": pl.Float64,
        "lon": pl.Float64,
    }
    return pl.DataFrame(rows, schema=schema)


def _layer(rows: list[dict[str, object]], with_object_id: bool = False) -> pl.DataFrame:
    if with_object_id:
        schema = {
            "object_id": pl.Utf8,
            "lat": pl.Float64,
            "lon": pl.Float64,
        }
    else:
        schema = {"lat": pl.Float64, "lon": pl.Float64}
    return pl.DataFrame(rows, schema=schema)


def _coord_at_offset(
    base_lat: float, base_lon: float, *, north_m: float = 0.0, east_m: float = 0.0
) -> tuple[float, float]:
    """Return a coordinate offset by ~`north_m` north and `east_m` east.

    Uses local flat-earth approximation: 1 deg lat ≈ 111 km,
    1 deg lon ≈ 111 km × cos(lat). Good enough for the few-hundred-m
    distances tested here.
    """
    import math

    dlat = north_m / 111_000.0
    dlon = east_m / (111_000.0 * math.cos(math.radians(base_lat)))
    return base_lat + dlat, base_lon + dlon


def test_adds_count_columns_per_layer_and_radius() -> None:
    objects = _objects(
        [{"object_id": "a", "lat": KAZAN_LAT, "lon": KAZAN_LON}]
    )
    far_lat, far_lon = _coord_at_offset(KAZAN_LAT, KAZAN_LON, north_m=2_000.0)
    near_lat, near_lon = _coord_at_offset(KAZAN_LAT, KAZAN_LON, east_m=200.0)
    stations = _layer([{"lat": near_lat, "lon": near_lon}, {"lat": far_lat, "lon": far_lon}])

    out = compute_object_zonal_features(
        objects, layers={"stations": stations}, radii_m=[100, 300, 500, 800]
    )

    expected_extra = {
        "stations_within_100m",
        "stations_within_300m",
        "stations_within_500m",
        "stations_within_800m",
    }
    assert expected_extra.issubset(set(out.columns))
    # 200m near point counts in 300/500/800 but not 100; 2 km point never counts.
    assert out["stations_within_100m"][0] == 0
    assert out["stations_within_300m"][0] == 1
    assert out["stations_within_500m"][0] == 1
    assert out["stations_within_800m"][0] == 1


def test_count_grows_monotonically_with_radius() -> None:
    base = (KAZAN_LAT, KAZAN_LON)
    objects = _objects([{"object_id": "a", "lat": base[0], "lon": base[1]}])

    near, mid, far_in, far_out = (
        _coord_at_offset(*base, east_m=80.0),     # within 100
        _coord_at_offset(*base, east_m=250.0),    # within 300
        _coord_at_offset(*base, east_m=700.0),    # within 800
        _coord_at_offset(*base, east_m=900.0),    # outside 800
    )
    layer = _layer([
        {"lat": near[0], "lon": near[1]},
        {"lat": mid[0], "lon": mid[1]},
        {"lat": far_in[0], "lon": far_in[1]},
        {"lat": far_out[0], "lon": far_out[1]},
    ])

    out = compute_object_zonal_features(
        objects, layers={"poi": layer}, radii_m=[100, 300, 500, 800]
    )

    assert out["poi_within_100m"][0] == 1
    assert out["poi_within_300m"][0] == 2
    assert out["poi_within_500m"][0] == 2
    assert out["poi_within_800m"][0] == 3


def test_multiple_layers_processed_independently() -> None:
    base = (KAZAN_LAT, KAZAN_LON)
    objects = _objects([{"object_id": "a", "lat": base[0], "lon": base[1]}])

    near = _coord_at_offset(*base, east_m=80.0)
    stations = _layer([{"lat": near[0], "lon": near[1]}])
    entrances = _layer([])

    out = compute_object_zonal_features(
        objects,
        layers={"stations": stations, "entrances": entrances},
        radii_m=[100, 500],
    )

    assert out["stations_within_100m"][0] == 1
    assert out["stations_within_500m"][0] == 1
    assert out["entrances_within_100m"][0] == 0
    assert out["entrances_within_500m"][0] == 0


def test_self_excluded_when_layer_carries_object_id() -> None:
    """An object that appears in its own layer (e.g. apartments → other
    apartments) must not count itself in any radius."""
    base = (KAZAN_LAT, KAZAN_LON)
    objects = _objects([
        {"object_id": "a", "lat": base[0], "lon": base[1]},
        {"object_id": "b", "lat": _coord_at_offset(*base, east_m=120.0)[0],
         "lon": _coord_at_offset(*base, east_m=120.0)[1]},
    ])
    # Layer == objects (same coordinates and ids)
    layer = _layer(
        [
            {"object_id": "a", "lat": base[0], "lon": base[1]},
            {"object_id": "b", "lat": _coord_at_offset(*base, east_m=120.0)[0],
             "lon": _coord_at_offset(*base, east_m=120.0)[1]},
        ],
        with_object_id=True,
    )

    out = compute_object_zonal_features(
        objects, layers={"apartments": layer}, radii_m=[300]
    )

    # a and b are 120 m apart → each sees the other within 300 m, not itself.
    assert out["apartments_within_300m"].to_list() == [1, 1]


def test_distinct_objects_at_same_coord_still_counted() -> None:
    """Two objects with different object_id but identical (lat, lon) — both
    are real objects, both should count toward each other's density."""
    base = (KAZAN_LAT, KAZAN_LON)
    objects = _objects([
        {"object_id": "a", "lat": base[0], "lon": base[1]},
        {"object_id": "b", "lat": base[0], "lon": base[1]},
    ])
    layer = _layer(
        [
            {"object_id": "a", "lat": base[0], "lon": base[1]},
            {"object_id": "b", "lat": base[0], "lon": base[1]},
        ],
        with_object_id=True,
    )

    out = compute_object_zonal_features(
        objects, layers={"layer": layer}, radii_m=[100]
    )

    # a sees b at distance 0 m (within 100); b sees a likewise. Self excluded.
    assert out["layer_within_100m"].to_list() == [1, 1]


def test_empty_objects_returns_empty_with_columns() -> None:
    objects = _objects([])
    layer = _layer([{"lat": KAZAN_LAT, "lon": KAZAN_LON}])

    out = compute_object_zonal_features(
        objects, layers={"poi": layer}, radii_m=[100, 500]
    )

    assert out.is_empty()
    assert "poi_within_100m" in out.columns
    assert "poi_within_500m" in out.columns


def test_empty_layer_yields_zero_counts() -> None:
    objects = _objects([{"object_id": "a", "lat": KAZAN_LAT, "lon": KAZAN_LON}])
    layer = _layer([])

    out = compute_object_zonal_features(
        objects, layers={"poi": layer}, radii_m=[100, 500]
    )

    assert out["poi_within_100m"][0] == 0
    assert out["poi_within_500m"][0] == 0


def test_no_layers_returns_input_unchanged() -> None:
    objects = _objects([{"object_id": "a", "lat": KAZAN_LAT, "lon": KAZAN_LON}])

    out = compute_object_zonal_features(
        objects, layers={}, radii_m=[100, 500]
    )

    assert out.columns == objects.columns
    assert out.height == objects.height


def test_radii_order_does_not_matter() -> None:
    base = (KAZAN_LAT, KAZAN_LON)
    objects = _objects([{"object_id": "a", "lat": base[0], "lon": base[1]}])
    near = _coord_at_offset(*base, east_m=200.0)
    layer = _layer([{"lat": near[0], "lon": near[1]}])

    out_a = compute_object_zonal_features(
        objects, layers={"poi": layer}, radii_m=[100, 300, 500, 800]
    )
    out_b = compute_object_zonal_features(
        objects, layers={"poi": layer}, radii_m=[800, 100, 500, 300]
    )

    for r in (100, 300, 500, 800):
        col = f"poi_within_{r}m"
        assert out_a[col][0] == out_b[col][0]


def test_haversine_threshold_is_strict_lt() -> None:
    """A point exactly at the radius is NOT included. Matches the
    convention used by compute_object_metro_features (`< R`)."""
    base = (KAZAN_LAT, KAZAN_LON)
    objects = _objects([{"object_id": "a", "lat": base[0], "lon": base[1]}])
    boundary = _coord_at_offset(*base, east_m=300.0)
    # Sanity: actual haversine is very close to 300.0 m.
    actual = haversine_meters(base[0], base[1], boundary[0], boundary[1])
    assert 290.0 < actual < 310.0
    layer = _layer([{"lat": boundary[0], "lon": boundary[1]}])

    threshold = round(actual) + 5
    out = compute_object_zonal_features(
        objects, layers={"poi": layer}, radii_m=[threshold]
    )
    col = f"poi_within_{threshold}m"
    assert out[col][0] == 1
