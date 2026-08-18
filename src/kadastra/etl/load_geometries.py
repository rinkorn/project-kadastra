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

import polars as pl
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


def load_geojsonseq_points(path_str: str) -> pl.DataFrame:
    """Read a GeoJSON-seq file and return one (lat, lon) per feature.

    Point geometries pass through unchanged; LineString / Polygon / Multi*
    are reduced to their centroid. Missing file → empty frame so downstream
    emits zero counts.
    """
    path = Path(path_str)
    if not path.is_file():
        return pl.DataFrame({"lat": [], "lon": []}, schema={"lat": pl.Float64, "lon": pl.Float64})
    lats: list[float] = []
    lons: list[float] = []
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("\x1e"):
                line = line.lstrip("\x1e").strip()
                if not line:
                    continue
            feature = json.loads(line)
            geom_dict = feature.get("geometry")
            if geom_dict is None:
                continue
            geom = shape(geom_dict)
            if geom.is_empty:
                continue
            pt = geom if geom.geom_type == "Point" else geom.centroid
            pt_x, pt_y = pt.coords[0]
            lons.append(float(pt_x))
            lats.append(float(pt_y))
    return pl.DataFrame({"lat": lats, "lon": lons})
