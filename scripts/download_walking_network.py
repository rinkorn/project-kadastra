"""Download an OSM walking-network dump for the Kazan agglomeration via Overpass.

The result is the raw Overpass JSON saved to disk; conversion into the
edges-table parquet that NetworkxRoadGraph reads is a separate step
(scripts/build_road_graph_artifact.py).

Default bbox covers the 30 km Kazan-agglomeration buffer from
ADR-0007 with some slack: ``55.5..56.05 lat, 48.7..49.55 lon``. The
query selects ``highway=*`` excluding motorways/trunks (pedestrians
don't walk on them and they distort travel-distance for everyone
else).

The script is idempotent: if the output file exists and ``--force``
is not passed, it exits without re-downloading. Polite to Overpass:
single connection, 600 s timeout, default endpoint
``https://overpass-api.de/api/interpreter`` (override via
``--endpoint``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx

_DEFAULT_BBOX = "55.5,48.7,56.05,49.55"  # south, west, north, east
_DEFAULT_OUT = Path("data/raw/osm/kazan_walking_network.json")
_DEFAULT_ENDPOINT = "https://overpass-api.de/api/interpreter"

_QUERY_TEMPLATE = """
[out:json][timeout:600];
way[\"highway\"]
   [\"highway\"!~\"^(motorway|trunk|motorway_link|trunk_link|construction|proposed)$\"]
   ({bbox});
out geom;
""".strip()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--bbox",
        default=_DEFAULT_BBOX,
        help="south,west,north,east (default: Kazan agglomeration ~30 km buffer)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUT,
        help=f"Output path for raw Overpass JSON (default: {_DEFAULT_OUT})",
    )
    p.add_argument(
        "--endpoint",
        default=_DEFAULT_ENDPOINT,
        help=f"Overpass endpoint (default: {_DEFAULT_ENDPOINT})",
    )
    p.add_argument("--force", action="store_true", help="Re-download even if output exists")
    args = p.parse_args()

    if args.out.exists() and not args.force:
        size_mb = args.out.stat().st_size / 1024 / 1024
        print(
            f"{args.out} already exists ({size_mb:.1f} MB); pass --force to re-download.",
            flush=True,
        )
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)

    query = _QUERY_TEMPLATE.format(bbox=args.bbox)
    print(f"POST {args.endpoint}  bbox={args.bbox}", flush=True)
    print(f"  expecting a few minutes; saving to {args.out}", flush=True)

    headers = {
        # Overpass returns 406 for the default httpx UA; use a contact-
        # bearing string per the Overpass usage guidelines.
        "User-Agent": ("kadastra-pilot/0.1 (https://github.com/joeblackdev/kadastra; rinkorn.alb@gmail.com)"),
        "Accept": "application/json,*/*",
    }
    with httpx.Client(timeout=httpx.Timeout(900.0), headers=headers) as client:
        try:
            r = client.post(args.endpoint, data={"data": query})
        except httpx.HTTPError as exc:
            sys.exit(f"Overpass request failed: {exc}")

    if r.status_code != 200:
        sys.exit(f"Overpass returned HTTP {r.status_code}: {r.text[:500]}")

    args.out.write_bytes(r.content)
    size_mb = len(r.content) / 1024 / 1024
    print(f"Saved {size_mb:.1f} MB to {args.out}", flush=True)


if __name__ == "__main__":
    main()
