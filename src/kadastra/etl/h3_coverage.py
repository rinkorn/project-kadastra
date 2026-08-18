from collections.abc import Iterable

import h3
from shapely.geometry.base import BaseGeometry


def geometry_to_h3_cells(geometry: BaseGeometry, resolution: int) -> set[str]:
    h3shape = h3.geo_to_h3shape(geometry)
    return set(h3.h3shape_to_cells(h3shape, resolution))


def h3_cells_to_latlng(cells: Iterable[str]) -> list[tuple[float, float]]:
    """Cell-centre coordinates as ``(lat, lon)`` in WGS84 degrees.

    A cell's reference point for location ЦОФ is its centroid — the
    point whose features a point object inherits when joined to the
    cell (ADR-0027).
    """
    return [h3.cell_to_latlng(cell) for cell in cells]
