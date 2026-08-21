"""Convert raw Overpass JSON into the edges-table parquet that the
:class:`NetworkxRoadGraph` adapter loads at boot.

Pure local processing: reads ``--src`` (Overpass JSON), runs
``build_road_graph_edges_from_overpass`` (edges + OSM tags), applies the
ADR-0030 water-crossing filter against ``--water`` (phantom crossings
over water polygons without a bridge/tunnel tag are dropped), writes the
result to ``--out`` (default: Settings.road_graph_edges_path).

When the water file is missing the filter is skipped with a warning —
the artifact is still built (idempotent on machines without raw OSM
extracts), but the phantom crossings then remain.

Use after ``scripts/download_walking_network.py`` produced the JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kadastra.config import Settings
from kadastra.etl.load_geometries import load_geojsonseq_geometries
from kadastra.etl.road_graph_edges_from_overpass import (
    build_road_graph_edges_from_overpass,
)
from kadastra.etl.road_graph_water_filter import (
    DEFAULT_MAX_WATER_CROSSING_M,
    filter_water_crossing_edges,
)

_DEFAULT_WATER = Path("data/raw/osm/kazan-agg-water.geojsonseq")


def main() -> None:
    settings = Settings()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--src",
        type=Path,
        default=Path("data/raw/osm/kazan_walking_network.json"),
        help="Path to raw Overpass JSON (from download_walking_network.py)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=settings.road_graph_edges_path,
        help=("Output parquet path (default: Settings.road_graph_edges_path)"),
    )
    p.add_argument(
        "--water",
        type=Path,
        default=_DEFAULT_WATER,
        help=f"Water polygons GeoJSON-seq for the ADR-0030 filter (default: {_DEFAULT_WATER})",
    )
    p.add_argument(
        "--max-water-crossing-m",
        type=float,
        default=DEFAULT_MAX_WATER_CROSSING_M,
        help=f"Water-crossing length threshold in meters (default: {DEFAULT_MAX_WATER_CROSSING_M})",
    )
    args = p.parse_args()

    if not args.src.is_file():
        raise SystemExit(f"--src does not exist: {args.src}. Run scripts/download_walking_network.py first.")

    print(f"Reading {args.src} ...", flush=True)
    payload = json.loads(args.src.read_text(encoding="utf-8"))
    n_elements = len(payload.get("elements", []))
    print(f"  {n_elements:,} elements in payload", flush=True)

    edges = build_road_graph_edges_from_overpass(payload)
    print(f"  -> {edges.height:,} edges", flush=True)

    if args.water.is_file():
        water_polygons = load_geojsonseq_geometries({"water": str(args.water)})["water"]
        filtered = filter_water_crossing_edges(
            edges,
            water_polygons,
            max_crossing_m=args.max_water_crossing_m,
        )
        dropped = edges.height - filtered.height
        print(
            f"  water filter ({args.water}): dropped {dropped:,} phantom crossings "
            f"(>{args.max_water_crossing_m:g} m over water, no bridge/tunnel)",
            flush=True,
        )
        edges = filtered
    else:
        print(f"  WARNING: {args.water} not found — water-crossing filter skipped", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    edges.write_parquet(args.out)
    size_mb = args.out.stat().st_size / 1024 / 1024
    print(f"Wrote {size_mb:.1f} MB to {args.out}", flush=True)


if __name__ == "__main__":
    main()
