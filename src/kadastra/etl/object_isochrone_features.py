"""15-minute walking-isochrone enrichment (ADR-0024, group 2).

For each H3 cell (``isochrone_cache_resolution``, res 11 by default)
the builder script runs a cutoff Dijkstra from the cell centre over the
OSM pedestrian graph (``15 мин × 80 м/мин = 1200 м``) and aggregates
what the walker reaches:

- ``iso15_pop_count``     (Float64) — population living in the res-11
  cells covered by the isochrone. First approximation sanctioned by
  ADR-0024: ОКТМО population (ADR-0022) distributed uniformly over the
  res-11 cells that contain valuation objects of that ОКТМО; cells
  without objects carry 0 (ADR-0024 «Аудит данных», п. 3).
- ``iso15_amenity_count`` (Int64) — number of point POIs (ADR-0019
  layers) reachable within the cutoff.
- ``iso15_metro_reach``   (Int64, 0/1) — whether at least one metro
  station (ADR-0011) is reachable within the cutoff.

``BuildObjectFeatures`` LEFT JOINs the cache via
:func:`join_isochrone_features` (object → its res-11 cell → cached
isochrone) after the RAW_OBJECT_SCHEMA reset — same recompute-from-
silver pattern as ADR-0022/0023.
"""

from __future__ import annotations

import h3
import polars as pl

from kadastra.ports.road_graph import RoadGraphPort

ISO15_FEATURE_COLUMNS: tuple[str, ...] = (
    "iso15_pop_count",
    "iso15_amenity_count",
    "iso15_metro_reach",
)

ISOCHRONE_CACHE_SCHEMA: dict[str, type[pl.DataType] | pl.DataType] = {
    "h3_index": pl.Utf8,
    "iso15_pop_count": pl.Float64,
    "iso15_amenity_count": pl.Int64,
    "iso15_metro_reach": pl.Int64,
}


def compute_isochrone_cache(
    cells: list[str],
    *,
    road_graph: RoadGraphPort,
    cutoff_m: float,
    resolution: int,
    poi_coords: list[tuple[float, float]],
    metro_coords: list[tuple[float, float]],
    cell_population: dict[str, float],
) -> pl.DataFrame:
    """Compute the isochrone feature row for each H3 cell.

    A POI/metro point counts as reached when its snapped node is
    reachable and ``source→node distance + point snap ≤ cutoff_m`` —
    the same end-to-end convention as ``distance_matrix_m``. The source
    cell itself is always part of its own isochrone, so a cell whose
    centre snaps farther than ``cutoff_m`` (off-graph area) still gets
    its own population, with zero amenities and ``metro_reach = 0``.

    ``cell_population`` maps a res-``resolution`` cell id to the
    population attributed to it (see the module docstring); cells absent
    from the map contribute 0.
    """
    if not cells:
        return pl.DataFrame(schema=ISOCHRONE_CACHE_SCHEMA)

    # Snap POI/metro sets once — node ids are stable for the graph
    # instance, so per-cell queries reuse them.
    poi_snaps = [road_graph.snap_node(coord) for coord in poi_coords]
    metro_snaps = [road_graph.snap_node(coord) for coord in metro_coords]

    rows: list[tuple[str, float, int, int]] = []
    for cell in cells:
        center = h3.cell_to_latlng(cell)
        reach = road_graph.reachable_nodes_within_m(center, cutoff_m)
        reach_cells = {cell}
        for node_id in reach:
            node_lat, node_lon = road_graph.node_coord(node_id)
            reach_cells.add(h3.latlng_to_cell(node_lat, node_lon, resolution))
        pop = sum(cell_population.get(c, 0.0) for c in reach_cells)
        amenity = sum(1 for node_id, snap_m in poi_snaps if node_id in reach and reach[node_id] + snap_m <= cutoff_m)
        metro = int(any(node_id in reach and reach[node_id] + snap_m <= cutoff_m for node_id, snap_m in metro_snaps))
        rows.append((cell, pop, amenity, metro))

    return pl.DataFrame(rows, schema=ISOCHRONE_CACHE_SCHEMA, orient="row")


def join_isochrone_features(
    objects: pl.DataFrame,
    cache: pl.DataFrame,
    *,
    resolution: int,
) -> pl.DataFrame:
    """LEFT JOIN the per-hex isochrone cache onto objects.

    Each object inherits the features of the res-``resolution`` cell its
    (lat, lon) falls into. Objects with null coordinates or with a cell
    missing from the cache get nulls. Idempotent: pre-existing feature
    columns are dropped first (the store is read-write; a rerun reads
    its own enriched output).
    """
    drop_existing = [c for c in ISO15_FEATURE_COLUMNS if c in objects.columns]
    if drop_existing:
        objects = objects.drop(drop_existing)

    null_columns = [
        pl.lit(None, dtype=pl.Float64).alias("iso15_pop_count"),
        pl.lit(None, dtype=pl.Int64).alias("iso15_amenity_count"),
        pl.lit(None, dtype=pl.Int64).alias("iso15_metro_reach"),
    ]
    if objects.height == 0:
        return objects.with_columns(null_columns)

    cells = [
        h3.latlng_to_cell(lat, lon, resolution) if lat is not None and lon is not None else None
        for lat, lon in zip(objects["lat"].to_list(), objects["lon"].to_list(), strict=True)
    ]
    with_index = objects.with_columns(pl.Series("h3_index", cells, dtype=pl.Utf8))
    if cache.is_empty():
        return with_index.with_columns(null_columns).drop("h3_index")

    right = cache.select(["h3_index", *ISO15_FEATURE_COLUMNS])
    return with_index.join(right, on="h3_index", how="left").drop("h3_index")
