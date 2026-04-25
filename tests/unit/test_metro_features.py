import h3
import polars as pl
import pytest

from kadastra.etl.metro_features import compute_metro_features

KAZAN_LAT, KAZAN_LON = 55.7887, 49.1221


def _coverage_for_cell(cell: str, resolution: int) -> pl.DataFrame:
    return pl.DataFrame({"h3_index": [cell], "resolution": [resolution]})


def test_compute_metro_features_returns_expected_columns() -> None:
    cell = h3.latlng_to_cell(KAZAN_LAT, KAZAN_LON, 8)
    coverage = _coverage_for_cell(cell, 8)
    stations = pl.DataFrame({"lat": [KAZAN_LAT], "lon": [KAZAN_LON]})
    entrances = pl.DataFrame({"lat": [KAZAN_LAT], "lon": [KAZAN_LON]})

    result = compute_metro_features(coverage, stations, entrances)

    assert set(result.columns) == {
        "h3_index",
        "resolution",
        "dist_metro_m",
        "dist_entrance_m",
        "count_stations_1km",
        "count_entrances_500m",
    }


def test_dist_metro_is_zero_when_station_sits_at_hex_center() -> None:
    cell = h3.latlng_to_cell(KAZAN_LAT, KAZAN_LON, 8)
    center_lat, center_lon = h3.cell_to_latlng(cell)
    coverage = _coverage_for_cell(cell, 8)
    stations = pl.DataFrame({"lat": [center_lat], "lon": [center_lon]})
    entrances = pl.DataFrame({"lat": [60.0], "lon": [40.0]})

    result = compute_metro_features(coverage, stations, entrances)

    assert result["dist_metro_m"][0] == pytest.approx(0.0, abs=1e-6)


def test_count_stations_within_1km_radius() -> None:
    cell = h3.latlng_to_cell(KAZAN_LAT, KAZAN_LON, 8)
    center_lat, center_lon = h3.cell_to_latlng(cell)
    coverage = _coverage_for_cell(cell, 8)
    # three near (< 1km) + one far away (~111 km)
    stations = pl.DataFrame(
        {
            "lat": [center_lat, center_lat + 0.001, center_lat - 0.001, center_lat + 1.0],
            "lon": [center_lon, center_lon, center_lon, center_lon],
        }
    )
    entrances = pl.DataFrame({"lat": [60.0], "lon": [40.0]})

    result = compute_metro_features(coverage, stations, entrances)

    assert result["count_stations_1km"][0] == 3


def test_count_entrances_within_500m_radius() -> None:
    cell = h3.latlng_to_cell(KAZAN_LAT, KAZAN_LON, 8)
    center_lat, center_lon = h3.cell_to_latlng(cell)
    coverage = _coverage_for_cell(cell, 8)
    stations = pl.DataFrame({"lat": [60.0], "lon": [40.0]})
    # two within 500m + two beyond
    entrances = pl.DataFrame(
        {
            "lat": [center_lat, center_lat + 0.001, center_lat + 0.01, center_lat + 0.02],
            "lon": [center_lon, center_lon, center_lon, center_lon],
        }
    )

    result = compute_metro_features(coverage, stations, entrances)

    assert result["count_entrances_500m"][0] == 2


def test_far_hex_has_no_metro_in_radius_and_large_distance() -> None:
    spb_cell = h3.latlng_to_cell(59.9343, 30.3351, 8)
    coverage = _coverage_for_cell(spb_cell, 8)
    stations = pl.DataFrame({"lat": [KAZAN_LAT], "lon": [KAZAN_LON]})
    entrances = pl.DataFrame({"lat": [KAZAN_LAT], "lon": [KAZAN_LON]})

    result = compute_metro_features(coverage, stations, entrances)

    assert result["count_stations_1km"][0] == 0
    assert result["count_entrances_500m"][0] == 0
    assert result["dist_metro_m"][0] > 1_000_000  # ~1100 km Kazan-SPB


def test_compute_metro_features_preserves_coverage_row_count() -> None:
    cells = [h3.latlng_to_cell(KAZAN_LAT + 0.01 * i, KAZAN_LON, 8) for i in range(5)]
    coverage = pl.DataFrame({"h3_index": cells, "resolution": [8] * 5})
    stations = pl.DataFrame({"lat": [KAZAN_LAT], "lon": [KAZAN_LON]})
    entrances = pl.DataFrame({"lat": [KAZAN_LAT], "lon": [KAZAN_LON]})

    result = compute_metro_features(coverage, stations, entrances)

    assert len(result) == 5
