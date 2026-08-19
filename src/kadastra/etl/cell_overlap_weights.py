"""ADR-0027 §12: area-share-weighted assignment of cell ЦОФ to objects.

An object whose footprint spans several res-N cells should inherit a
**blend** of those cells' location features, weighted by the share of
the object's area that falls in each cell — not the features of the
single central cell. This module computes the per-object ``{h3_index,
weight}`` mapping once; ``BuildObjectFeatures`` reuses it across all
six Слой 1 feature sets (A1 of the overlap-weighting plan).

Areas are measured in UTM-39N (EPSG:32639) — the same projection the
share/distance feature pipelines already use — so weights are free of
the latitude-dependent distortion a WGS84 area would introduce. Objects
without a usable geometry (null/empty ``polygon_wkt_3857``) produce no
rows here; the caller falls back to single-cell-by-centroid for them.
"""

from __future__ import annotations

import h3
import polars as pl
import shapely
from pyproj import Transformer
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform

# web-mercator metres (storage CRS of polygon_wkt_3857) → WGS84 degrees
# (H3 cell boundaries live in lat/lon). Mirrors the transformer in
# api/routes.py; kept local so the ETL layer has no API dependency.
_3857_TO_4326 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
# WGS84 degrees → UTM-39N metres — area projection (same as the
# object_geom_distance / object_polygon feature pipelines).
_4326_TO_UTM = Transformer.from_crs("EPSG:4326", "EPSG:32639", always_xy=True)


def _reproject(geom: BaseGeometry, transformer: Transformer) -> BaseGeometry:
    return shapely_transform(lambda x, y, z=None: transformer.transform(x, y), geom)


def _cell_polygon(cell: str) -> BaseGeometry:
    """H3 cell boundary as a Shapely polygon in WGS84 (lon, lat).

    ``h3.cell_to_boundary`` returns ``(lat, lng)`` tuples; Shapely
    expects ``(x=lng, y=lat)`` so we swap each pair and close the ring.
    """
    boundary = h3.cell_to_boundary(cell)
    ring = [(lng, lat) for lat, lng in boundary]
    ring.append(ring[0])
    return shapely.geometry.Polygon(ring)


def object_cell_weights(geom_wkt_3857: str | None, resolution: int) -> list[tuple[str, float]]:
    """``[(h3_index, weight), …]`` for every ``resolution`` cell the
    object's footprint overlaps, weights summing to 1.0.

    Null / empty / unparseable geometry → ``[]`` (caller falls back to
    the centroid cell). Weights are the UTM-39N area of the
    object∩cell intersection, normalized over all overlapping cells.
    """
    if not geom_wkt_3857:
        return []
    try:
        geom_merc = shapely.from_wkt(geom_wkt_3857)
    except Exception:
        return []
    if geom_merc.is_empty:
        return []

    geom_wgs = _reproject(geom_merc, _3857_TO_4326)
    candidates = h3.h3shape_to_cells(h3.geo_to_h3shape(geom_wgs), resolution)
    if not candidates:
        return []

    geom_utm = _reproject(geom_wgs, _4326_TO_UTM)
    areas: list[tuple[str, float]] = []
    for cell in candidates:
        cell_utm = _reproject(_cell_polygon(cell), _4326_TO_UTM)
        inter = geom_utm.intersection(cell_utm)
        a = inter.area
        if a > 0:
            areas.append((cell, a))
    total = sum(a for _, a in areas)
    if total <= 0:
        return []
    return [(cell, a / total) for cell, a in areas]


def compute_overlap_weights(df: pl.DataFrame, *, resolution: int) -> pl.DataFrame:
    """Frame ``(object_id, h3_index, weight)`` for every object×cell pair.

    Reads ``object_id`` and ``polygon_wkt_3857`` from ``df``. Objects
    with no usable geometry contribute a single centroid-cell row with
    weight 1.0 (the single-cell fallback), so downstream joins never
    drop them. Empty ``df`` → empty frame with the same schema.
    """
    schema = {"object_id": pl.Utf8, "h3_index": pl.Utf8, "weight": pl.Float64}
    if df.is_empty():
        return pl.DataFrame(schema=schema)

    rows: list[tuple[str, str, float]] = []
    for obj_id, wkt, lat, lon in zip(
        df["object_id"].to_list(),
        df["polygon_wkt_3857"].to_list(),
        df["lat"].to_list(),
        df["lon"].to_list(),
        strict=False,
    ):
        weights = object_cell_weights(wkt, resolution)
        if not weights:
            # No geometry → single centroid cell, weight 1.0.
            rows.append((obj_id, h3.latlng_to_cell(float(lat), float(lon), resolution), 1.0))
        else:
            rows.extend((obj_id, cell, w) for cell, w in weights)
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(
        {
            "object_id": [r[0] for r in rows],
            "h3_index": [r[1] for r in rows],
            "weight": [r[2] for r in rows],
        },
        schema=schema,
    )
