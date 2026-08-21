"""Download an OSM walking-network dump for the Kazan agglomeration via Overpass.

The result is the raw Overpass JSON saved to disk; conversion into the
edges-table parquet that NetworkxRoadGraph reads is a separate step
(scripts/build_road_graph_artifact.py).

Default bbox covers the 30 km Kazan-agglomeration buffer from
ADR-0007 **plus ~12 km of extra slack** (ADR-0030): ``55.39..56.16 lat,
48.51..49.74 lon``. The graph is cut by this bbox, so objects near the
agglomeration edge used to get inflated "route to the cut" distances;
the extra ring keeps the full walking path inside the graph for any
object we score.

The query selects ``highway=*`` excluding motorways/trunk (pedestrians
don't walk on them and they distort travel-distance for everyone else),
in union with all ``bridge=yes`` ways regardless of class (ADR-0030) —
the class filter used to drop real bridges with a pedestrian part
(Millennium bridge is ``highway=trunk``). Motorways and motorway_links
stay excluded even on bridges: pedestrians never walk there.

The script is idempotent: if the output file exists and ``--force``
is not passed, it exits without re-downloading. Polite to Overpass:
single connection, 600 s timeout, default endpoint
``https://overpass-api.de/api/interpreter`` (override via
``--endpoint``). Transient failures (429/5xx, e.g. the 504 Overpass
returns under load) are retried with exponential backoff.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import httpx

# south, west, north, east — ADR-0007 buffer + ~12 km slack (ADR-0030):
# 12 km ≈ 0.108° lat, ≈ 0.192° lon at Kazan latitude.
_DEFAULT_BBOX = "55.39,48.51,56.16,49.74"
_DEFAULT_OUT = Path("data/raw/osm/kazan_walking_network.json")
_DEFAULT_ENDPOINT = "https://overpass-api.de/api/interpreter"

# Overpass answers 504/429 under load; the widened bbox + bridge union
# (ADR-0030) makes the query heavier, so retry instead of failing the
# whole rebuild run.
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
_MAX_ATTEMPTS = 5
_INITIAL_BACKOFF_S = 60.0

_QUERY_TEMPLATE = """
[out:json][timeout:600];
(
  way[\"highway\"]
     [\"highway\"!~\"^(motorway|trunk|motorway_link|trunk_link|construction|proposed)$\"]
     ({bbox});
  way[\"highway\"][\"bridge\"=\"yes\"]
     [\"highway\"!~\"^(motorway|motorway_link|construction|proposed)$\"]
     ({bbox});
);
out geom;
""".strip()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--bbox",
        default=_DEFAULT_BBOX,
        help="south,west,north,east (default: Kazan agglomeration 30 km buffer + 12 km slack, ADR-0030)",
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
    r: httpx.Response | None = None
    delay = _INITIAL_BACKOFF_S
    with httpx.Client(timeout=httpx.Timeout(900.0), headers=headers) as client:
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                r = client.post(args.endpoint, data={"data": query})
            except httpx.HTTPError as exc:
                reason = str(exc)
            else:
                if r.status_code == 200:
                    break
                reason = f"HTTP {r.status_code}: {r.text[:200]}"
                if r.status_code not in _RETRYABLE_STATUS:
                    sys.exit(f"Overpass returned {reason}")
                r = None
            if attempt == _MAX_ATTEMPTS:
                sys.exit(f"Overpass request failed after {_MAX_ATTEMPTS} attempts: {reason}")
            print(
                f"  attempt {attempt}/{_MAX_ATTEMPTS} failed ({reason}); retrying in {delay:.0f}s",
                flush=True,
            )
            time.sleep(delay)
            delay *= 2

    assert r is not None
    args.out.write_bytes(r.content)
    size_mb = len(r.content) / 1024 / 1024
    print(f"Saved {size_mb:.1f} MB to {args.out}", flush=True)


if __name__ == "__main__":
    main()
