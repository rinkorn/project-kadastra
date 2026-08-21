"""Build the per-hex 15-min walking-isochrone cache (ADR-0024, group 2).

For every res-``Settings.isochrone_cache_resolution`` H3 cell that
contains at least one valuation object, runs a cutoff Dijkstra over the
OSM pedestrian graph (``isochrone_walking_time_min`` ×
``isochrone_walking_speed_m_per_min`` = 15 × 80 = 1200 m by default)
from the cell centre and aggregates:

- ``iso15_pop_count`` — population of the reachable cells. First
  approximation (ADR-0024 «Аудит данных», п. 3): ОКТМО population from
  the gold ``oktmo_population`` (ADR-0022) is distributed uniformly over
  the res-11 cells that contain valuation objects of that ОКТМО; cells
  without objects carry 0.
- ``iso15_amenity_count`` — point POIs of the ADR-0019 layers
  (``walk_dist_layer_names``) reachable within the cutoff.
- ``iso15_metro_reach`` — 1 if a metro station (ADR-0011) is reachable.

Only cells with objects are cached (78k vs 1.24M coverage cells — the
cache exists to be LEFT JOINed from objects, so object-less cells would
never be read); objects outside the cache get nulls downstream.

Output:

    data/silver/isochrone_cache/region={code}/h3_p={res}/data.parquet
      h3_index, iso15_pop_count, iso15_amenity_count, iso15_metro_reach

The Dijkstra work is pure-Python (networkx), so parallelism is
process-based: each worker loads its own graph copy in the Pool
initializer (spawn-safe) and computes a chunk of cells.

Запуск:
    uv run python scripts/build_isochrone_cache_per_hex.py
"""

from __future__ import annotations

import io
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import h3
import polars as pl

from kadastra.adapters.networkx_road_graph import NetworkxRoadGraph
from kadastra.adapters.parquet_valuation_object_store import ParquetValuationObjectStore
from kadastra.composition_root import Container
from kadastra.config import Settings
from kadastra.domain.asset_class import AssetClass
from kadastra.etl.load_geometries import load_geojsonseq_points
from kadastra.etl.object_isochrone_features import compute_isochrone_cache

# --- worker state (set once per process by the Pool initializer) -------

_GRAPH: NetworkxRoadGraph | None = None
_CUTOFF_M: float = 0.0
_RESOLUTION: int = 11
_POI_COORDS: list[tuple[float, float]] = []
_METRO_COORDS: list[tuple[float, float]] = []
_CELL_POPULATION: dict[str, float] = {}


def _init_worker(
    edges_path: str,
    cutoff_m: float,
    resolution: int,
    poi_coords: list[tuple[float, float]],
    metro_coords: list[tuple[float, float]],
    cell_population: dict[str, float],
) -> None:
    global _GRAPH, _CUTOFF_M, _RESOLUTION, _POI_COORDS, _METRO_COORDS, _CELL_POPULATION
    _GRAPH = NetworkxRoadGraph.from_parquet(Path(edges_path))
    _CUTOFF_M = cutoff_m
    _RESOLUTION = resolution
    _POI_COORDS = poi_coords
    _METRO_COORDS = metro_coords
    _CELL_POPULATION = cell_population


def _compute_chunk(cells: list[str]) -> pl.DataFrame:
    assert _GRAPH is not None, "worker not initialized"
    return compute_isochrone_cache(
        cells,
        road_graph=_GRAPH,
        cutoff_m=_CUTOFF_M,
        resolution=_RESOLUTION,
        poi_coords=_POI_COORDS,
        metro_coords=_METRO_COORDS,
        cell_population=_CELL_POPULATION,
    )


# --- main-thread assembly -----------------------------------------------


def load_objects(settings: Settings) -> pl.DataFrame:
    store = ParquetValuationObjectStore(settings.valuation_object_store_path)
    frames = []
    for ac in AssetClass:
        df = store.load(settings.region_code, ac)
        if not df.is_empty():
            frames.append(df.select(["lat", "lon", "oktmo_full", "oktmo_population"]))
    if not frames:
        sys.exit("no valuation objects found — run assemble/build pipeline first")
    return pl.concat(frames)


def build_cell_population(objects: pl.DataFrame, resolution: int) -> dict[str, float]:
    """ОКТМО population → uniform over the res-``resolution`` cells that
    contain valuation objects of that ОКТМО (ADR-0024, п. 3)."""
    with_cell = objects.with_columns(
        pl.struct(["lat", "lon"])
        .map_elements(
            lambda r: (
                h3.latlng_to_cell(r["lat"], r["lon"], resolution)
                if r["lat"] is not None and r["lon"] is not None
                else None
            ),
            return_dtype=pl.Utf8,
        )
        .alias("h3_index")
    ).drop_nulls(["h3_index", "oktmo_full", "oktmo_population"])

    oktmo_pop = with_cell.group_by("oktmo_full").agg(pl.col("oktmo_population").first())
    oktmo_cells = with_cell.group_by("oktmo_full").agg(pl.col("h3_index").n_unique().alias("n_cells"))
    stats = oktmo_pop.join(oktmo_cells, on="oktmo_full").with_columns(
        (pl.col("oktmo_population") / pl.col("n_cells")).alias("pop_per_cell")
    )

    cell_oktmo = with_cell.select(["h3_index", "oktmo_full"]).unique()
    cell_pop = cell_oktmo.join(stats.select(["oktmo_full", "pop_per_cell"]), on="oktmo_full")
    return dict(zip(cell_pop["h3_index"].to_list(), cell_pop["pop_per_cell"].to_list(), strict=True))


def load_poi_coords(settings: Settings) -> list[tuple[float, float]]:
    coords: list[tuple[float, float]] = []
    for name in settings.walk_dist_layer_names:
        path = settings.geom_distance_layer_paths.get(name)
        if path is None:
            continue
        df = load_geojsonseq_points(path)
        coords.extend((float(lat), float(lon)) for lat, lon in df.iter_rows())
    return coords


def load_metro_coords(settings: Settings, container: Container) -> list[tuple[float, float]]:
    raw = container.build_s3_raw_data()
    stations = pl.read_csv(io.BytesIO(raw.read_bytes(settings.metro_stations_key)))
    return [(float(lat), float(lon)) for lat, lon in stations.select(["lat", "lon"]).iter_rows()]


def main() -> int:
    started = time.monotonic()
    settings = Settings()
    container = Container(settings)
    resolution = settings.isochrone_cache_resolution
    cutoff_m = settings.isochrone_walking_time_min * settings.isochrone_walking_speed_m_per_min

    objects = load_objects(settings)
    cells = sorted(
        {
            h3.latlng_to_cell(lat, lon, resolution)
            for lat, lon in objects.select(["lat", "lon"]).iter_rows()
            if lat is not None and lon is not None
        }
    )
    print(f"=> object cells res{resolution}: {len(cells):,} (from {objects.height:,} objects)", flush=True)

    cell_population = build_cell_population(objects, resolution)
    print(f"=> cells with population attribution: {len(cell_population):,}", flush=True)

    poi_coords = load_poi_coords(settings)
    metro_coords = load_metro_coords(settings, container)
    print(f"=> POIs: {len(poi_coords):,}  metro stations: {len(metro_coords):,}  cutoff: {cutoff_m:.0f} m", flush=True)

    workers = min(10, max(1, (os.cpu_count() or 2) - 2))
    n_chunks = workers * 8
    chunk_size = max(1, (len(cells) + n_chunks - 1) // n_chunks)
    chunks = [cells[i : i + chunk_size] for i in range(0, len(cells), chunk_size)]
    print(f"=> workers: {workers}  chunks: {len(chunks)} × ~{chunk_size}", flush=True)

    ctx = mp.get_context("spawn")
    parts: list[pl.DataFrame] = []
    with ctx.Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(
            str(settings.road_graph_edges_path),
            cutoff_m,
            resolution,
            poi_coords,
            metro_coords,
            cell_population,
        ),
    ) as pool:
        for i, part in enumerate(pool.imap(_compute_chunk, chunks), start=1):
            parts.append(part)
            if i % 8 == 0 or i == len(chunks):
                elapsed = time.monotonic() - started
                print(f"   chunk {i}/{len(chunks)}  elapsed {elapsed:.0f}s", flush=True)

    cache = pl.concat(parts)
    out_dir = settings.isochrone_cache_path / f"region={settings.region_code}" / f"h3_p={resolution}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "data.parquet"
    cache.write_parquet(out_path)
    print(f"=> wrote {out_path}  rows={cache.height:,}  total {time.monotonic() - started:.0f}s", flush=True)

    print("\n=> isochrone stats:")
    pop_col, amen_col, metro_col = "iso15_pop_count", "iso15_amenity_count", "iso15_metro_reach"
    print(f"   {pop_col}:     mean={cache[pop_col].mean():.0f}  p50={cache[pop_col].median():.0f}")
    print(f"   {amen_col}: mean={cache[amen_col].mean():.1f}  p50={cache[amen_col].median():.0f}")
    print(f"   {metro_col}:   {cache[metro_col].mean():.1%} of cells")
    return 0


if __name__ == "__main__":
    sys.exit(main())
