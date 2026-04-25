"""Pure functions to flatten one NSPD GeoJSON feature into a row dict.

Each input is a single ``Feature`` as it appears on disk in
``data/raw/nspd/{buildings,landplots}-kazan/page-NNNN.json`` (the value
of ``data["data"]["features"][i]``). The output is a flat ``dict``
ready to be appended to a polars frame:

- atomic options (cad_num, area, cost_value, cost_index, ...) are
  unwrapped from ``properties.options``,
- centroid is computed in EPSG:3857 and projected to WGS84,
- the original polygon is preserved as WKT (in EPSG:3857) for any
  subsequent spatial filtering that doesn't want to redo the projection.
"""

from __future__ import annotations

from typing import Any

from kadastra.domain.classify_nspd_purpose import classify_nspd_building_purpose


def parse_nspd_building_feature(feature: dict[str, Any]) -> dict[str, Any]:
    raise NotImplementedError


def parse_nspd_landplot_feature(feature: dict[str, Any]) -> dict[str, Any]:
    raise NotImplementedError
