from pathlib import Path

from shapely.geometry.base import BaseGeometry


class LocalGeoJsonRegionBoundary:
    def __init__(self, path: Path, region_code_field: str = "shapeISO") -> None:
        self._path = path
        self._region_code_field = region_code_field

    def get_boundary(self, region_code: str) -> BaseGeometry:
        raise NotImplementedError
