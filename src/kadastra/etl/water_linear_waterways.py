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

from typing import Any

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


def parse_width_m(raw: object) -> float | None:
    """Parse an OSM ``width`` tag into meters. ``None`` when unusable."""
    raise NotImplementedError


def buffer_linear_waterway_features(
    features: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Replace linear waterway features with their buffered polygons.

    LineString/MultiLineString features whose ``waterway`` property is one
    of ``LINEAR_WATERWAY_CLASSES`` are buffered (see module docstring);
    every other feature passes through unchanged.
    """
    raise NotImplementedError
