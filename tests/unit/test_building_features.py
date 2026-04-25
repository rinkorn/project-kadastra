import h3
import polars as pl
import pytest

from kadastra.etl.building_features import compute_building_features

KAZAN_LAT, KAZAN_LON = 55.7887, 49.1221


def _coverage(cells: list[str], resolution: int) -> pl.DataFrame:
    return pl.DataFrame({"h3_index": cells, "resolution": [resolution] * len(cells)})


def _buildings(rows: list[dict[str, object]]) -> pl.DataFrame:
    schema = {
        "lat": pl.Float64,
        "lon": pl.Float64,
        "building": pl.Utf8,
        "levels": pl.Utf8,
        "flats": pl.Utf8,
    }
    return pl.DataFrame(rows, schema=schema)


def test_returns_expected_columns() -> None:
    cell = h3.latlng_to_cell(KAZAN_LAT, KAZAN_LON, 8)
    coverage = _coverage([cell], 8)
    buildings = _buildings(
        [{"lat": KAZAN_LAT, "lon": KAZAN_LON, "building": "house", "levels": "2", "flats": None}]
    )

    result = compute_building_features(coverage, buildings)

    assert set(result.columns) >= {
        "h3_index",
        "resolution",
        "building_count",
        "building_count_apartments",
        "building_count_detached",
        "levels_mean",
        "flats_total",
    }


def test_counts_buildings_per_hex() -> None:
    cell = h3.latlng_to_cell(KAZAN_LAT, KAZAN_LON, 8)
    coverage = _coverage([cell], 8)
    buildings = _buildings(
        [
            {"lat": KAZAN_LAT, "lon": KAZAN_LON, "building": "house", "levels": "2", "flats": None},
            {"lat": KAZAN_LAT, "lon": KAZAN_LON, "building": "apartments", "levels": "9", "flats": "120"},
            {"lat": KAZAN_LAT, "lon": KAZAN_LON, "building": "detached", "levels": "1", "flats": None},
        ]
    )

    result = compute_building_features(coverage, buildings)

    row = result.row(0, named=True)
    assert row["building_count"] == 3
    assert row["building_count_apartments"] == 1
    assert row["building_count_detached"] == 1


def test_levels_mean_uses_only_non_null_levels() -> None:
    cell = h3.latlng_to_cell(KAZAN_LAT, KAZAN_LON, 8)
    coverage = _coverage([cell], 8)
    buildings = _buildings(
        [
            {"lat": KAZAN_LAT, "lon": KAZAN_LON, "building": "house", "levels": "2", "flats": None},
            {"lat": KAZAN_LAT, "lon": KAZAN_LON, "building": "house", "levels": "4", "flats": None},
            {"lat": KAZAN_LAT, "lon": KAZAN_LON, "building": "house", "levels": None, "flats": None},
        ]
    )

    result = compute_building_features(coverage, buildings)

    assert result["levels_mean"][0] == pytest.approx(3.0)


def test_flats_total_sums_non_null_flats() -> None:
    cell = h3.latlng_to_cell(KAZAN_LAT, KAZAN_LON, 8)
    coverage = _coverage([cell], 8)
    buildings = _buildings(
        [
            {"lat": KAZAN_LAT, "lon": KAZAN_LON, "building": "apartments", "levels": "9", "flats": "100"},
            {"lat": KAZAN_LAT, "lon": KAZAN_LON, "building": "apartments", "levels": "10", "flats": "50"},
            {"lat": KAZAN_LAT, "lon": KAZAN_LON, "building": "house", "levels": "2", "flats": None},
        ]
    )

    result = compute_building_features(coverage, buildings)

    assert result["flats_total"][0] == 150


def test_empty_hex_gets_zero_counts() -> None:
    populated = h3.latlng_to_cell(KAZAN_LAT, KAZAN_LON, 8)
    empty = h3.latlng_to_cell(60.0, 30.0, 8)  # SPB area, no buildings here
    coverage = _coverage([populated, empty], 8)
    buildings = _buildings(
        [{"lat": KAZAN_LAT, "lon": KAZAN_LON, "building": "house", "levels": "2", "flats": None}]
    )

    result = compute_building_features(coverage, buildings)

    by_hex = {row["h3_index"]: row for row in result.iter_rows(named=True)}
    assert by_hex[populated]["building_count"] == 1
    assert by_hex[empty]["building_count"] == 0
    assert by_hex[empty]["building_count_apartments"] == 0


def test_handles_invalid_levels_strings_gracefully() -> None:
    cell = h3.latlng_to_cell(KAZAN_LAT, KAZAN_LON, 8)
    coverage = _coverage([cell], 8)
    # OSM data sometimes has "2-3" or "approx" — should not crash, treated as null
    buildings = _buildings(
        [
            {"lat": KAZAN_LAT, "lon": KAZAN_LON, "building": "house", "levels": "2", "flats": None},
            {"lat": KAZAN_LAT, "lon": KAZAN_LON, "building": "house", "levels": "garbage", "flats": None},
        ]
    )

    result = compute_building_features(coverage, buildings)

    assert result["building_count"][0] == 2
    assert result["levels_mean"][0] == pytest.approx(2.0)


def test_raises_on_mixed_resolutions() -> None:
    coverage = pl.DataFrame(
        {
            "h3_index": [
                h3.latlng_to_cell(KAZAN_LAT, KAZAN_LON, 7),
                h3.latlng_to_cell(KAZAN_LAT, KAZAN_LON, 8),
            ],
            "resolution": [7, 8],
        }
    )
    buildings = _buildings([])

    with pytest.raises(ValueError, match="single resolution"):
        compute_building_features(coverage, buildings)
