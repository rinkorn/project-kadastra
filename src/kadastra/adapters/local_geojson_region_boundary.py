import json
from pathlib import Path
from typing import Any, cast

from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry


class LocalGeoJsonRegionBoundary:
    def __init__(self, path: Path, region_code_field: str = "shapeISO") -> None:
        self._path = path
        self._region_code_field = region_code_field

    def get_boundary(self, region_code: str) -> BaseGeometry:
        with self._path.open() as fh:
            data = cast(dict[str, Any], json.load(fh))
        for feature in data.get("features", []):
            properties = feature.get("properties") or {}
            if properties.get(self._region_code_field) == region_code:
                return shape(feature["geometry"])
        raise KeyError(f"Region {region_code!r} not found in {self._path}")
