"""Pure-function ETL: Overpass JSON → road graph edges DataFrame.

The output DataFrame is the persistence format consumed by
:class:`NetworkxRoadGraph.from_parquet`. Schema:
``(from_lat, from_lon, to_lat, to_lon, length_m)``.

Each Overpass `way` element with a ``geometry`` list of N nodes is
expanded into N-1 consecutive edges; non-way elements (nodes,
relations) and ways without geometry are skipped.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from kadastra.etl.haversine import haversine_meters


_OUTPUT_SCHEMA = {
    "from_lat": pl.Float64,
    "from_lon": pl.Float64,
    "to_lat": pl.Float64,
    "to_lon": pl.Float64,
    "length_m": pl.Float64,
}


def build_road_graph_edges_from_overpass(
    payload: dict[str, Any],
) -> pl.DataFrame:
    rows: list[dict[str, float]] = []
    for element in payload.get("elements", []):
        if element.get("type") != "way":
            continue
        geometry = element.get("geometry")
        if not geometry or len(geometry) < 2:
            continue
        for prev, curr in zip(geometry, geometry[1:], strict=False):
            from_lat = float(prev["lat"])
            from_lon = float(prev["lon"])
            to_lat = float(curr["lat"])
            to_lon = float(curr["lon"])
            rows.append(
                {
                    "from_lat": from_lat,
                    "from_lon": from_lon,
                    "to_lat": to_lat,
                    "to_lon": to_lon,
                    "length_m": haversine_meters(
                        from_lat, from_lon, to_lat, to_lon
                    ),
                }
            )

    if not rows:
        return pl.DataFrame(schema=_OUTPUT_SCHEMA)
    return pl.DataFrame(rows, schema=_OUTPUT_SCHEMA)
