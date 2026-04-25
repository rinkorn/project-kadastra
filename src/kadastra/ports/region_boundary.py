from typing import Protocol

from shapely.geometry.base import BaseGeometry


class RegionBoundaryPort(Protocol):
    def get_boundary(self, region_code: str) -> BaseGeometry: ...
