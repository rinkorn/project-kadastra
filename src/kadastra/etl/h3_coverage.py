import h3
from shapely.geometry.base import BaseGeometry


def geometry_to_h3_cells(geometry: BaseGeometry, resolution: int) -> set[str]:
    h3shape = h3.geo_to_h3shape(geometry)
    return set(h3.h3shape_to_cells(h3shape, resolution))
