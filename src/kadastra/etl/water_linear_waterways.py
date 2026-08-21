"""Buffer linear OSM waterways (``waterway=river|stream|canal|ditch|drain``
ways) into polygons for the water layer (ADR-0032).

OSM maps big water bodies as polygons (``natural=water`` /
``waterway=riverbank``), but most in-city rivers, streams and channels
exist only as linear ``waterway=*`` ways. Without buffering they never
reach ``kazan-agg-water.geojsonseq``: rivers are invisible on the map,
``dist_to_water_m`` / ``water_share_*`` understate water, and the
road-graph water filter (ADR-0030) misses crossings over linear rivers.

The buffer half-width comes from the way's ``width`` tag when it parses
to a positive number of meters (values like ``"10"``, ``"10.5"`` or
``"6.5 m"``); otherwise a documented per-class approximation from
``DEFAULT_HALF_WIDTHS_M`` is used. Buffering runs in UTM 39N
(EPSG:32639, ≤ 0.1 % length distortion at Kazan latitude) with round
caps/joins — natural for river channels — and the polygon is projected
back to WGS84.
"""

from __future__ import annotations

import re
from typing import Any

from pyproj import Transformer
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform

# Waterway classes extracted as linear ways into the water layer.
LINEAR_WATERWAY_CLASSES: tuple[str, ...] = ("river", "stream", "canal", "ditch", "drain")

# Approximate half-widths (m) per waterway class, used only when the way
# carries no usable ``width`` tag. These are deliberately rough: typical
# in-city channel widths in the Kazan agglomeration, biased to the wide
# side so the road-graph water filter (30 m crossing threshold, ADR-0030)
# still sees narrow streams while ``water_share_*`` isn't flooded by
# oversized polygons.
DEFAULT_HALF_WIDTHS_M: dict[str, float] = {
    "river": 10.0,
    "canal": 8.0,
    "stream": 3.0,
    "ditch": 1.5,
    "drain": 1.5,
}

# Provenance marker on buffered features, mirroring the ESA-augment
# pattern (``source=esa_*``) so the polygons synthesized from lines are
# distinguishable from native OSM water polygons downstream.
BUFFERED_SOURCE = "osm_waterway_line_buffer"

# Segments per quarter circle in the round caps/joins — shapely default.
_BUFFER_QUAD_SEGS = 8

# Leading unsigned number, optionally with a decimal comma, optionally
# followed by a unit ("10", "10.5", "10,5", "6.5 m", "6.5m").
_WIDTH_RE = re.compile(r"^\s*(\d+(?:[.,]\d+)?)")

_TO_UTM = Transformer.from_crs("EPSG:4326", "EPSG:32639", always_xy=True)
_TO_WGS84 = Transformer.from_crs("EPSG:32639", "EPSG:4326", always_xy=True)


def parse_width_m(raw: object) -> float | None:
    """Parse an OSM ``width`` tag into meters. ``None`` when unusable."""
    if not isinstance(raw, str):
        return None
    match = _WIDTH_RE.match(raw)
    if match is None:
        return None
    value = float(match.group(1).replace(",", "."))
    return value if value > 0 else None


def _half_width_m(properties: dict[str, Any]) -> float | None:
    """Half width in meters: ``width/2`` when the tag parses, else the
    per-class approximation. ``None`` for classes without a documented
    default — those are left as lines rather than buffered at a made-up
    width."""
    width_m = parse_width_m(properties.get("width"))
    if width_m is not None:
        return width_m / 2.0
    waterway = properties.get("waterway")
    if not isinstance(waterway, str):
        return None
    return DEFAULT_HALF_WIDTHS_M.get(waterway)


def _buffer_line(geom: BaseGeometry, half_width_m: float) -> BaseGeometry:
    """Buffer a WGS84 line in UTM 39N, round caps/joins, back to WGS84."""
    utm = shapely_transform(lambda x, y, z=None: _TO_UTM.transform(x, y), geom)
    buffered = utm.buffer(
        half_width_m,
        quad_segs=_BUFFER_QUAD_SEGS,
        cap_style="round",
        join_style="round",
    )
    return shapely_transform(lambda x, y, z=None: _TO_WGS84.transform(x, y), buffered)


def buffer_linear_waterway_features(
    features: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Replace linear waterway features with their buffered polygons.

    LineString/MultiLineString features whose ``waterway`` property is one
    of ``LINEAR_WATERWAY_CLASSES`` are buffered (see module docstring);
    every other feature passes through unchanged.
    """
    out: list[dict[str, Any]] = []
    for feature in features:
        geom_dict = feature.get("geometry") or {}
        if geom_dict.get("type") not in ("LineString", "MultiLineString"):
            out.append(feature)
            continue
        properties: dict[str, Any] = feature.get("properties") or {}
        half_width = _half_width_m(properties)
        if half_width is None:
            out.append(feature)
            continue
        polygon = _buffer_line(shape(geom_dict), half_width)
        out.append(
            {
                **feature,
                "properties": {**properties, "source": BUFFERED_SOURCE},
                "geometry": mapping(polygon),
            }
        )
    return out
