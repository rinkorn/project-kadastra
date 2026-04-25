import polars as pl

from kadastra.etl.object_metro_features import compute_object_metro_features

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


def _points(rows: list[dict[str, float]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema={"lat": pl.Float64, "lon": pl.Float64})


def test_appends_distance_and_count_columns() -> None:
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
    stations = _points([{"lat": KAZAN_LAT, "lon": KAZAN_LON}])
    entrances = _points([{"lat": KAZAN_LAT, "lon": KAZAN_LON}])

    result = compute_object_metro_features(objects, stations, entrances)

    expected_extra = {
        "dist_metro_m",
        "dist_entrance_m",
        "count_stations_1km",
        "count_entrances_500m",
    }
    assert expected_extra.issubset(set(result.columns))
    assert result["dist_metro_m"][0] < 1.0
    assert result["count_stations_1km"][0] == 1
    assert result["count_entrances_500m"][0] == 1


def test_distant_object_has_high_distance_and_zero_counts() -> None:
    far_lat, far_lon = KAZAN_LAT + 0.5, KAZAN_LON + 0.5  # ~50 km away
    objects = _objects(
        [
            {
                "object_id": "way/1",
                "asset_class": "house",
                "lat": far_lat,
                "lon": far_lon,
                "levels": None,
                "flats": None,
            }
        ]
    )
    stations = _points([{"lat": KAZAN_LAT, "lon": KAZAN_LON}])
    entrances = _points([{"lat": KAZAN_LAT, "lon": KAZAN_LON}])

    result = compute_object_metro_features(objects, stations, entrances)

    assert result["dist_metro_m"][0] > 10_000
    assert result["count_stations_1km"][0] == 0
    assert result["count_entrances_500m"][0] == 0


def test_preserves_object_columns_and_row_order() -> None:
    objects = _objects(
        [
            {
                "object_id": "way/1",
                "asset_class": "apartment",
                "lat": KAZAN_LAT,
                "lon": KAZAN_LON,
                "levels": 9,
                "flats": 72,
            },
            {
                "object_id": "way/2",
                "asset_class": "house",
                "lat": KAZAN_LAT + 0.01,
                "lon": KAZAN_LON,
                "levels": 1,
                "flats": None,
            },
        ]
    )
    stations = _points([{"lat": KAZAN_LAT, "lon": KAZAN_LON}])
    entrances = _points([{"lat": KAZAN_LAT, "lon": KAZAN_LON}])

    result = compute_object_metro_features(objects, stations, entrances)

    assert result["object_id"].to_list() == ["way/1", "way/2"]
    assert {"object_id", "asset_class", "lat", "lon", "levels", "flats"}.issubset(
        set(result.columns)
    )


def test_empty_objects_returns_empty_with_feature_columns() -> None:
    objects = _objects([])
    stations = _points([{"lat": KAZAN_LAT, "lon": KAZAN_LON}])
    entrances = _points([{"lat": KAZAN_LAT, "lon": KAZAN_LON}])

    result = compute_object_metro_features(objects, stations, entrances)

    assert result.is_empty()
    for col in (
        "dist_metro_m",
        "dist_entrance_m",
        "count_stations_1km",
        "count_entrances_500m",
    ):
        assert col in result.columns


def test_no_stations_yields_inf_distance_and_zero_counts() -> None:
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
    empty_points = _points([])

    result = compute_object_metro_features(objects, empty_points, empty_points)

    assert result["count_stations_1km"][0] == 0
    assert result["count_entrances_500m"][0] == 0
    # No reference points → use a sentinel that signals "unknown / never close":
    # finite but very large, so downstream models treat it as "far".
    assert result["dist_metro_m"][0] > 1e6
    assert result["dist_entrance_m"][0] > 1e6
