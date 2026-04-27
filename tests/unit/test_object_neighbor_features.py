import polars as pl

from kadastra.etl.object_neighbor_features import compute_object_neighbor_features

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


def _row(oid: str, ac: str, lat: float, lon: float) -> dict[str, object]:
    return {
        "object_id": oid,
        "asset_class": ac,
        "lat": lat,
        "lon": lon,
        "levels": None,
        "flats": None,
    }


def test_appends_count_columns_per_class() -> None:
    objects = _objects([_row("way/1", "apartment", KAZAN_LAT, KAZAN_LON)])

    result = compute_object_neighbor_features(objects, radius_m=500)

    for col in (
        "count_apartments_500m",
        "count_houses_500m",
        "count_commercial_500m",
    ):
        assert col in result.columns


def test_counts_exclude_self() -> None:
    objects = _objects([_row("way/1", "apartment", KAZAN_LAT, KAZAN_LON)])

    result = compute_object_neighbor_features(objects, radius_m=500)

    assert result["count_apartments_500m"][0] == 0
    assert result["count_houses_500m"][0] == 0
    assert result["count_commercial_500m"][0] == 0


def test_counts_only_within_radius_and_per_class() -> None:
    objects = _objects(
        [
            _row("way/1", "apartment", KAZAN_LAT, KAZAN_LON),
            # ~100 m offsets, 3 neighbors of mixed class within 500m
            _row("way/2", "apartment", KAZAN_LAT + 0.0009, KAZAN_LON),
            _row("way/3", "house", KAZAN_LAT, KAZAN_LON + 0.0009),
            _row("way/4", "commercial", KAZAN_LAT - 0.0009, KAZAN_LON),
            # ~5 km away — outside radius
            _row("way/5", "apartment", KAZAN_LAT + 0.05, KAZAN_LON),
        ]
    )

    result = compute_object_neighbor_features(objects, radius_m=500).filter(pl.col("object_id") == "way/1")

    assert result["count_apartments_500m"][0] == 1  # way/2 only
    assert result["count_houses_500m"][0] == 1  # way/3
    assert result["count_commercial_500m"][0] == 1  # way/4


def test_radius_controls_inclusion() -> None:
    objects = _objects(
        [
            _row("way/1", "apartment", KAZAN_LAT, KAZAN_LON),
            _row("way/2", "apartment", KAZAN_LAT + 0.003, KAZAN_LON),  # ~330 m away
        ]
    )

    inside = compute_object_neighbor_features(objects, radius_m=500).filter(pl.col("object_id") == "way/1")
    outside = compute_object_neighbor_features(objects, radius_m=100).filter(pl.col("object_id") == "way/1")

    assert inside["count_apartments_500m"][0] == 1
    assert outside["count_apartments_500m"][0] == 0


def test_empty_objects_returns_empty_with_columns() -> None:
    result = compute_object_neighbor_features(_objects([]), radius_m=500)

    assert result.is_empty()
    for col in (
        "count_apartments_500m",
        "count_houses_500m",
        "count_commercial_500m",
    ):
        assert col in result.columns


def test_unknown_asset_class_does_not_break_counting() -> None:
    objects = _objects(
        [
            _row("way/1", "apartment", KAZAN_LAT, KAZAN_LON),
            _row("way/2", "land_plot", KAZAN_LAT + 0.0009, KAZAN_LON),
        ]
    )

    result = compute_object_neighbor_features(objects, radius_m=500).filter(pl.col("object_id") == "way/1")

    assert result["count_apartments_500m"][0] == 0
    assert result["count_houses_500m"][0] == 0
    assert result["count_commercial_500m"][0] == 0
