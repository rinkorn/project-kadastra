"""Per-object geometric ZOFs from polygon_wkt_3857 (ADR-0018).

Reads ``polygon_wkt_3857`` (EPSG:3857 WKT) and appends 7 columns:

- ``polygon_area_m2``         (Float64) — shapely.area
- ``polygon_perimeter_m``     (Float64) — shapely.length
- ``polygon_compactness``     (Float64) — Polsby–Popper 4·π·A / P²
- ``polygon_convexity``       (Float64) — A / area(convex_hull)
- ``bbox_aspect_ratio``       (Float64) — long / short side of min rotated bbox
- ``polygon_orientation_deg`` (Float64) — long-axis angle of min rotated
                                          bbox, [0°, 180°)
- ``polygon_n_vertices``      (Int64)   — exterior vertex count

Null WKT → all 7 features null on that row.
"""

from __future__ import annotations

import polars as pl


def compute_object_geometry_features(objects: pl.DataFrame) -> pl.DataFrame:
    raise NotImplementedError
