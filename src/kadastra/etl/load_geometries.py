"""Load OSM-extracted GeoJSON-seq layers into Shapely geometries.

Shared by the per-object and per-cell feature builders so both read the
same layer files the same way. Geometry-agnostic: returns whatever
``shape()`` produces from each feature (Point / LineString / Polygon and
Multi* variants). Missing files yield empty layers — downstream produces
zero/null columns rather than failing while OSM extractions are still
being run.
"""

from __future__ import annotations

import json
from pathlib import Path

from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry


def load_geojsonseq_geometries(paths: dict[str, str]) -> dict[str, list[BaseGeometry]]:
    """Load each ``{name: path}`` GeoJSON-seq into a geometries list."""
    layers: dict[str, list[BaseGeometry]] = {}
    for name, path_str in paths.items():
        path = Path(path_str)
        if not path.is_file():
            layers[name] = []
            continue
        geoms: list[BaseGeometry] = []
        with path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("\x1e"):
                    line = line.lstrip("\x1e").strip()
                    if not line:
                        continue
                feature = json.loads(line)
                geom = feature.get("geometry")
                if geom is None:
                    continue
                geoms.append(shape(geom))
        layers[name] = geoms
    return layers
