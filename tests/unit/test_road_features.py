from typing import Any

import h3
import polars as pl
import pytest

from kadastra.etl.haversine import haversine_meters
from kadastra.etl.road_features import compute_road_features

KAZAN_LAT, KAZAN_LON = 55.7887, 49.1221


def _coverage(cells: list[str], resolution: int) -> pl.DataFrame:
    return pl.DataFrame({"h3_index": cells, "resolution": [resolution] * len(cells)})


def _way(coords: list[tuple[float, float]], **tags: str) -> dict[str, Any]:
    return {
        "tags": dict(tags),
        "geometry": [{"lat": lat, "lon": lon} for lat, lon in coords],
    }


def test_returns_expected_columns() -> None:
    cell = h3.latlng_to_cell(KAZAN_LAT, KAZAN_LON, 8)
    coverage = _coverage([cell], 8)

    result = compute_road_features(coverage, ways=[])

    assert set(result.columns) >= {"h3_index", "resolution", "road_length_m"}


def test_hex_without_roads_has_zero_length() -> None:
    cell = h3.latlng_to_cell(KAZAN_LAT, KAZAN_LON, 8)
    coverage = _coverage([cell], 8)

    result = compute_road_features(coverage, ways=[])

    assert result["road_length_m"][0] == 0.0


def test_short_segment_inside_one_hex_contributes_its_length() -> None:
    cell = h3.latlng_to_cell(KAZAN_LAT, KAZAN_LON, 8)
    center_lat, center_lon = h3.cell_to_latlng(cell)
    coverage = _coverage([cell], 8)
    # Two points very close to center; both fall into same hex
    p1 = (center_lat, center_lon)
    p2 = (center_lat + 0.0001, center_lon)  # ~11 m north
    way = _way([p1, p2], highway="primary")

    result = compute_road_features(coverage, ways=[way])

    expected = haversine_meters(p1[0], p1[1], p2[0], p2[1])
    assert result["road_length_m"][0] == pytest.approx(expected, rel=1e-9)


def test_multi_segment_way_sums_lengths_in_same_hex() -> None:
    cell = h3.latlng_to_cell(KAZAN_LAT, KAZAN_LON, 8)
    center_lat, center_lon = h3.cell_to_latlng(cell)
    coverage = _coverage([cell], 8)
    coords = [
        (center_lat, center_lon),
        (center_lat + 0.0001, center_lon),
        (center_lat + 0.0002, center_lon),
    ]
    way = _way(coords, highway="secondary")

    result = compute_road_features(coverage, ways=[way])

    expected = haversine_meters(*coords[0], *coords[1]) + haversine_meters(*coords[1], *coords[2])
    assert result["road_length_m"][0] == pytest.approx(expected, rel=1e-9)


def test_way_with_segments_in_adjacent_hexes_assigns_length_to_both() -> None:
    cell_a = h3.latlng_to_cell(KAZAN_LAT, KAZAN_LON, 8)
    a_lat, a_lon = h3.cell_to_latlng(cell_a)
    cell_b = next(c for c in h3.grid_disk(cell_a, 1) if c != cell_a)
    b_lat, b_lon = h3.cell_to_latlng(cell_b)
    coverage = _coverage([cell_a, cell_b], 8)

    eps = 0.00001
    way = _way(
        [
            (a_lat - eps, a_lon),
            (a_lat + eps, a_lon),
            (b_lat - eps, b_lon),
            (b_lat + eps, b_lon),
        ]
    )

    result = compute_road_features(coverage, ways=[way])

    by_hex = {row["h3_index"]: row["road_length_m"] for row in result.iter_rows(named=True)}
    assert by_hex[cell_a] > 0
    assert by_hex[cell_b] > 0


def test_distant_hex_gets_zero_when_road_is_elsewhere() -> None:
    populated = h3.latlng_to_cell(KAZAN_LAT, KAZAN_LON, 8)
    distant = h3.latlng_to_cell(60.0, 30.0, 8)
    coverage = _coverage([populated, distant], 8)
    way = _way(
        [(KAZAN_LAT, KAZAN_LON), (KAZAN_LAT + 0.0001, KAZAN_LON)], highway="primary"
    )

    result = compute_road_features(coverage, ways=[way])

    by_hex = {row["h3_index"]: row["road_length_m"] for row in result.iter_rows(named=True)}
    assert by_hex[populated] > 0
    assert by_hex[distant] == 0.0


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

    with pytest.raises(ValueError, match="single resolution"):
        compute_road_features(coverage, ways=[])
