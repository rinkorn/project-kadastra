from shapely.geometry.base import BaseGeometry


def geometry_to_h3_cells(geometry: BaseGeometry, resolution: int) -> set[str]:
    raise NotImplementedError
