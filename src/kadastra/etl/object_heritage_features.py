"""Per-object cultural-heritage (ОКН) features (ADR-0025, п. 2).

Source reality (ADR-0025 «Аудит данных»): the Минкульт open-data API is
unreachable from our network, so the layer is an OSM extract
(``data/raw/osm/kazan-agg-heritage.geojsonseq``, backup on S3). Only
~188 ОКН carry a ``heritage=*`` tag in the agglomeration — modest
coverage, recorded as a data-quality limitation in the ADR.

Two functions:

- :func:`parse_heritage_geojsonseq` — raw GeoJSON-seq → silver frame
  (``data/silver/heritage/region={code}/data.parquet``), filtering out
  the service/non-ОКН rows the extract also carries. Closed LineString
  building footprints are promoted to Polygons.
- :func:`compute_object_heritage_features` — per-object features:
  ``is_heritage_object`` (within 50 m of an ОКН — buffer match replaces
  the unavailable cad_num match), ``dist_to_nearest_heritage_m``,
  ``count_heritage_500m``, ``inside_heritage_zone`` (polygon containment
  over ОКН footprints; distance fallback < 100 m when the layer has no
  polygons — ADR-0025 «Открытые вопросы»).

All spatial work in EPSG:32639 (UTM-39N), same as the other
distance-feature pipelines.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import polars as pl
import shapely
import shapely.wkt
from pyproj import Transformer
from shapely.geometry import Polygon, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform

if TYPE_CHECKING:
    from collections.abc import Iterable

# UTM-39N — same projection as the other distance-feature pipelines.
_TO_UTM = Transformer.from_crs("EPSG:4326", "EPSG:32639", always_xy=True)

HERITAGE_SILVER_SCHEMA: dict[str, type[pl.DataType] | pl.DataType] = {
    "osm_id": pl.Utf8,
    "ref_egrokn": pl.Utf8,
    "heritage_level": pl.Utf8,
    "name": pl.Utf8,
    "lat": pl.Float64,
    "lon": pl.Float64,
    # WGS84 WKT for polygonal features (MultiPolygon / closed-LineString
    # footprints); null for point-only ОКН.
    "polygon_wkt": pl.Utf8,
}

HERITAGE_FEATURE_COLUMNS: tuple[str, ...] = (
    "is_heritage_object",
    "dist_to_nearest_heritage_m",
    "count_heritage_500m",
    "inside_heritage_zone",
)


def _to_polygon(geom: BaseGeometry) -> BaseGeometry | None:
    """Return the polygonal form of a heritage geometry, if any.

    OSM closed ways arrive as LineStrings in the extract; a ring with
    ≥4 vertices and matching endpoints is a building footprint.
    """
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom
    if geom.geom_type == "LineString":
        coords = list(geom.coords)
        if len(coords) >= 4 and coords[0] == coords[-1]:
            return Polygon(coords)
    return None


def parse_heritage_geojsonseq(lines: Iterable[str]) -> pl.DataFrame:
    """Parse GeoJSON-seq lines into the silver heritage frame.

    Keeps only features with a non-null ``properties.heritage`` tag.
    Returns an empty frame with :data:`HERITAGE_SILVER_SCHEMA` when no
    ОКН rows are present.
    """
    rows: list[dict[str, Any]] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("\x1e"):
            line = line.lstrip("\x1e").strip()
            if not line:
                continue
        feature = json.loads(line)
        props = feature.get("properties") or {}
        heritage_level = props.get("heritage")
        if not heritage_level:
            continue
        geom_dict = feature.get("geometry")
        if geom_dict is None:
            continue
        geom = shape(geom_dict)
        if geom.is_empty:
            continue
        centroid = geom.centroid
        polygon = _to_polygon(geom)
        rows.append(
            {
                "osm_id": str(feature.get("id")),
                "ref_egrokn": props.get("ref:egrokn"),
                "heritage_level": str(heritage_level),
                "name": props.get("name"),
                "lat": float(centroid.y),
                "lon": float(centroid.x),
                "polygon_wkt": polygon.wkt if polygon is not None else None,
            }
        )
    return pl.DataFrame(rows, schema=HERITAGE_SILVER_SCHEMA)


def _null_feature_columns(objects: pl.DataFrame) -> pl.DataFrame:
    return objects.with_columns(
        [
            pl.lit(None, dtype=pl.Int64).alias("is_heritage_object"),
            pl.lit(None, dtype=pl.Float64).alias("dist_to_nearest_heritage_m"),
            pl.lit(None, dtype=pl.Int64).alias("count_heritage_500m"),
            pl.lit(None, dtype=pl.Int64).alias("inside_heritage_zone"),
        ]
    )


def _project_lonlat(geom: BaseGeometry) -> BaseGeometry:
    return shapely_transform(lambda x, y, z=None: _TO_UTM.transform(x, y), geom)


def compute_object_heritage_features(
    objects: pl.DataFrame,
    *,
    heritage: pl.DataFrame,
    object_buffer_m: float = 50.0,
    count_radius_m: float = 500.0,
    zone_fallback_dist_m: float = 100.0,
) -> pl.DataFrame:
    """Append the 4 heritage feature columns to ``objects``.

    ``objects`` must carry ``lat``/``lon``; ``heritage`` follows
    :data:`HERITAGE_SILVER_SCHEMA`. An empty heritage layer yields
    all-null feature columns. Objects with null coordinates get nulls.
    Flag columns are Int64 (0/1) so the model feature selector picks
    them up as numerics.
    """
    n = objects.height
    if n == 0 or heritage.is_empty():
        return _null_feature_columns(objects)

    # ОКН centroids → UTM points (one STRtree for nearest + count).
    h_lats = heritage["lat"].to_numpy()
    h_lons = heritage["lon"].to_numpy()
    h_valid = ~(np.isnan(h_lats) | np.isnan(h_lons))
    if not h_valid.any():
        return _null_feature_columns(objects)
    h_xs, h_ys = _TO_UTM.transform(h_lons[h_valid], h_lats[h_valid])
    heritage_points = cast(np.ndarray[Any, Any], shapely.points(np.asarray(h_xs), np.asarray(h_ys)))

    dist_values: list[float | None] = [None] * n
    count_values: list[int | None] = [None] * n
    is_object_values: list[int | None] = [None] * n
    inside_values: list[int | None] = [None] * n

    obj_lats = objects["lat"].to_numpy()
    obj_lons = objects["lon"].to_numpy()
    valid = ~(np.isnan(obj_lats) | np.isnan(obj_lons))
    if valid.any():
        obj_xs, obj_ys = _TO_UTM.transform(obj_lons[valid], obj_lats[valid])
        points = cast(np.ndarray[Any, Any], shapely.points(np.asarray(obj_xs), np.asarray(obj_ys)))
        valid_idx = np.flatnonzero(valid)

        tree = shapely.STRtree(heritage_points)
        nearest_idx = cast(np.ndarray[Any, Any], tree.nearest(points))
        distances = shapely.distance(points, heritage_points[nearest_idx])
        pairs = tree.query(points, predicate="dwithin", distance=count_radius_m)
        counts = np.bincount(pairs[0], minlength=points.size) if pairs.size else np.zeros(points.size, dtype=np.int64)
        for pos, row_i in enumerate(valid_idx.tolist()):
            d = float(distances[pos])
            dist_values[row_i] = d
            count_values[row_i] = int(counts[pos])
            is_object_values[row_i] = 1 if d <= object_buffer_m else 0

        # inside_heritage_zone: polygon containment over ОКН footprints
        # when the layer carries polygons; distance fallback otherwise.
        polygons = [
            _project_lonlat(shapely.wkt.loads(wkt_str))
            for wkt_str in heritage["polygon_wkt"].to_list()
            if wkt_str is not None
        ]
        if polygons:
            poly_arr = np.asarray(polygons, dtype=object)
            hits = shapely.STRtree(poly_arr).query(points, predicate="intersects")
            inside_mask = np.zeros(points.size, dtype=bool)
            if hits.size:
                inside_mask[np.unique(hits[0])] = True
            for pos, row_i in enumerate(valid_idx.tolist()):
                inside_values[row_i] = 1 if inside_mask[pos] else 0
        else:
            for pos, row_i in enumerate(valid_idx.tolist()):
                inside_values[row_i] = 1 if float(distances[pos]) < zone_fallback_dist_m else 0

    return objects.with_columns(
        [
            pl.Series("is_heritage_object", is_object_values, dtype=pl.Int64),
            pl.Series("dist_to_nearest_heritage_m", dist_values, dtype=pl.Float64),
            pl.Series("count_heritage_500m", count_values, dtype=pl.Int64),
            pl.Series("inside_heritage_zone", inside_values, dtype=pl.Int64),
        ]
    )
