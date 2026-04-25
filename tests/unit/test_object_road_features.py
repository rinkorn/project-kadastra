import polars as pl

from kadastra.etl.object_road_features import compute_object_road_features

KAZAN_LAT, KAZAN_LON = 55.7887, 49.1221


def _objects(rows: list[dict[str, object]]) -> pl.DataFrame:
    schema = {
        "object_id": pl.Utf8,
        "asset_class": pl.Utf8,
        "lat": pl.Float64,
        "lon": pl.Float64,
        "levels": pl.Int64,
        "flats": pl.Int64,
    }
    return pl.DataFrame(rows, schema=schema)


def _way(coords: list[tuple[float, float]]) -> dict[str, object]:
    return {"geometry": [{"lat": lat, "lon": lon} for lat, lon in coords]}


def test_appends_road_length_column() -> None:
    objects = _objects(
        [
            {
                "object_id": "way/1",
                "asset_class": "apartment",
                "lat": KAZAN_LAT,
                "lon": KAZAN_LON,
                "levels": 9,
                "flats": 72,
            }
        ]
    )
    # 100m segment running through the object
    nearby_road = _way([(KAZAN_LAT, KAZAN_LON), (KAZAN_LAT + 0.0009, KAZAN_LON)])
    result = compute_object_road_features(objects, [nearby_road], radius_m=500)

    assert "road_length_500m" in result.columns
    # ~100m segment counted (100 ± 1)
    length = result["road_length_500m"][0]
    assert 90 < length < 110


def test_distant_road_not_counted() -> None:
    objects = _objects(
        [
            {
                "object_id": "way/1",
                "asset_class": "house",
                "lat": KAZAN_LAT,
                "lon": KAZAN_LON,
                "levels": 1,
                "flats": None,
            }
        ]
    )
    far_road = _way(
        [(KAZAN_LAT + 0.5, KAZAN_LON + 0.5), (KAZAN_LAT + 0.5, KAZAN_LON + 0.501)]
    )
    result = compute_object_road_features(objects, [far_road], radius_m=500)

    assert result["road_length_500m"][0] == 0.0


def test_empty_roads_yields_zero_length() -> None:
    objects = _objects(
        [
            {
                "object_id": "way/1",
                "asset_class": "apartment",
                "lat": KAZAN_LAT,
                "lon": KAZAN_LON,
                "levels": None,
                "flats": None,
            }
        ]
    )
    result = compute_object_road_features(objects, [], radius_m=500)

    assert result["road_length_500m"][0] == 0.0


def test_empty_objects_returns_empty_with_column() -> None:
    objects = _objects([])
    result = compute_object_road_features(objects, [], radius_m=500)

    assert result.is_empty()
    assert "road_length_500m" in result.columns


def test_radius_controls_inclusion() -> None:
    objects = _objects(
        [
            {
                "object_id": "way/1",
                "asset_class": "apartment",
                "lat": KAZAN_LAT,
                "lon": KAZAN_LON,
                "levels": None,
                "flats": None,
            }
        ]
    )
    # Segment ~250 m north of object
    moderate_road = _way(
        [(KAZAN_LAT + 0.0023, KAZAN_LON), (KAZAN_LAT + 0.0023, KAZAN_LON + 0.001)]
    )
    inside = compute_object_road_features(objects, [moderate_road], radius_m=500)
    outside = compute_object_road_features(objects, [moderate_road], radius_m=100)

    assert inside["road_length_500m"][0] > 0
    assert outside["road_length_500m"][0] == 0.0


def test_segment_length_double_counted_only_when_close() -> None:
    objects = _objects(
        [
            {
                "object_id": "way/1",
                "asset_class": "apartment",
                "lat": KAZAN_LAT,
                "lon": KAZAN_LON,
                "levels": None,
                "flats": None,
            }
        ]
    )
    way = _way(
        [
            (KAZAN_LAT, KAZAN_LON),
            (KAZAN_LAT + 0.0009, KAZAN_LON),  # ~100 m segment 1 (close)
            (KAZAN_LAT + 0.0018, KAZAN_LON),  # ~100 m segment 2 (close)
        ]
    )
    result = compute_object_road_features(objects, [way], radius_m=500)

    # Two ~100 m segments → ~200 m total
    assert 180 < result["road_length_500m"][0] < 220
