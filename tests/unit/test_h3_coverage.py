import h3
from shapely.geometry import MultiPolygon, Point

from kadastra.etl.h3_coverage import geometry_to_h3_cells

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
