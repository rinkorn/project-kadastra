"""Convert raw Overpass JSON into the edges-table parquet that the
:class:`NetworkxRoadGraph` adapter loads at boot.

Pure local processing: reads ``--src`` (Overpass JSON), runs
``build_road_graph_edges_from_overpass``, writes the result to
``--out`` (default: Settings.road_graph_edges_path).

Use after ``scripts/download_walking_network.py`` produced the JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kadastra.config import Settings
from kadastra.etl.road_graph_edges_from_overpass import (
    build_road_graph_edges_from_overpass,
)


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
    args = p.parse_args()

    if not args.src.is_file():
        raise SystemExit(f"--src does not exist: {args.src}. Run scripts/download_walking_network.py first.")

    print(f"Reading {args.src} ...", flush=True)
    payload = json.loads(args.src.read_text(encoding="utf-8"))
    n_elements = len(payload.get("elements", []))
    print(f"  {n_elements:,} elements in payload", flush=True)

    edges = build_road_graph_edges_from_overpass(payload)
    print(f"  -> {edges.height:,} edges", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    edges.write_parquet(args.out)
    size_mb = args.out.stat().st_size / 1024 / 1024
    print(f"Wrote {size_mb:.1f} MB to {args.out}", flush=True)


if __name__ == "__main__":
    main()
