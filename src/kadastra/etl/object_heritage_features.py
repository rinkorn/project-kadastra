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

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Iterable

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


def parse_heritage_geojsonseq(lines: Iterable[str]) -> pl.DataFrame:
    """Parse GeoJSON-seq lines into the silver heritage frame.

    Keeps only features with a non-null ``properties.heritage`` tag.
    Returns an empty frame with :data:`HERITAGE_SILVER_SCHEMA` when no
    ОКН rows are present.
    """
    raise NotImplementedError


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
    raise NotImplementedError
