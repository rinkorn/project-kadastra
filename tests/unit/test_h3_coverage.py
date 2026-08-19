import h3
import polars as pl
from shapely.geometry import MultiPolygon, Point

from kadastra.etl.h3_coverage import add_h3_index, geometry_to_h3_cells, h3_cells_to_latlng

KAZAN_LAT, KAZAN_LON = 55.7887, 49.1221
CHELNY_LAT, CHELNY_LON = 55.7430, 52.4112


def test_geometry_to_h3_cells_returns_cells_at_requested_resolution() -> None:
    polygon = Point(KAZAN_LON, KAZAN_LAT).buffer(0.01)

    cells = geometry_to_h3_cells(polygon, resolution=10)

    assert len(cells) > 0
    for cell in cells:
        assert h3.get_resolution(cell) == 10
        assert h3.is_valid_cell(cell)


def test_geometry_to_h3_cells_covers_centroid() -> None:
    polygon = Point(KAZAN_LON, KAZAN_LAT).buffer(0.01)

    cells = geometry_to_h3_cells(polygon, resolution=8)

    centroid_cell = h3.latlng_to_cell(KAZAN_LAT, KAZAN_LON, 8)
    assert centroid_cell in cells


def test_geometry_to_h3_cells_returns_a_set() -> None:
    polygon = Point(KAZAN_LON, KAZAN_LAT).buffer(0.005)

    cells = geometry_to_h3_cells(polygon, resolution=10)

    assert isinstance(cells, set)


def test_geometry_to_h3_cells_handles_multipolygon() -> None:
    p_kazan = Point(KAZAN_LON, KAZAN_LAT).buffer(0.005)
    p_chelny = Point(CHELNY_LON, CHELNY_LAT).buffer(0.005)
    multi = MultiPolygon([p_kazan, p_chelny])

    cells = geometry_to_h3_cells(multi, resolution=8)

    assert h3.latlng_to_cell(KAZAN_LAT, KAZAN_LON, 8) in cells
    assert h3.latlng_to_cell(CHELNY_LAT, CHELNY_LON, 8) in cells


def test_h3_cells_to_latlng_returns_centres() -> None:
    cell = h3.latlng_to_cell(KAZAN_LAT, KAZAN_LON, 10)

    centres = h3_cells_to_latlng([cell])

    assert len(centres) == 1
    lat, lon = centres[0]
    assert -90.0 <= lat <= 90.0
    assert -180.0 <= lon <= 180.0


def test_h3_cells_to_latlng_centre_is_close_to_source_point() -> None:
    cell = h3.latlng_to_cell(KAZAN_LAT, KAZAN_LON, 10)

    lat, lon = h3_cells_to_latlng([cell])[0]

    # res=10 cell is ~75 m across; the centroid must land within the
    # cell, i.e. within ~0.001° of the source point.
    assert abs(lat - KAZAN_LAT) < 0.001
    assert abs(lon - KAZAN_LON) < 0.001


def test_add_h3_index_assigns_cells_from_latlon() -> None:
    df = pl.DataFrame({"lat": [KAZAN_LAT], "lon": [KAZAN_LON]})

    out = add_h3_index(df, resolution=10)

    assert out["h3_index"][0] == h3.latlng_to_cell(KAZAN_LAT, KAZAN_LON, 10)
