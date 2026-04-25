"""Extract polygon layers (water/park/industrial/cemetery) from the
agglomeration-clipped OSM PBF into per-layer GeoJSON-seq files.

The output is consumed by ``compute_object_polygon_features`` as input
for poly-area buffer features (see ADR-0014).

Source: ``data/raw/osm/kazan-agg.osm.pbf`` (generated earlier by the
buildings pipeline, see ADR-0007).

Outputs (one per layer): ``data/raw/osm/kazan-agg-{layer}.geojsonseq``

Each layer is defined by an ``osmium tags-filter`` expression. We then
``osmium export -f geojsonseq`` so each line is one GeoJSON Feature with
geometry + properties — easy to stream-read in Python.

The script is idempotent: skips any layer whose output already exists,
unless ``--force`` is passed.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# OSM tag filter expressions, one per layer.
# Format follows osmium-tool's `tags-filter` rules: `<object-types>/<key=value>`.
_LAYER_FILTERS: dict[str, list[str]] = {
    "water": [
        "wa/natural=water",
        "wa/waterway=riverbank",
        "wa/landuse=reservoir",
    ],
    "park": [
        "wa/leisure=park",
        "wa/leisure=garden",
        "wa/landuse=forest",
    ],
    "industrial": [
        "wa/landuse=industrial",
        "wa/landuse=brownfield",
    ],
    "cemetery": [
        "wa/landuse=cemetery",
        "wa/amenity=grave_yard",
    ],
}

_DEFAULT_SRC = Path("data/raw/osm/kazan-agg.osm.pbf")
_OUT_DIR = Path("data/raw/osm")


def _run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        sys.exit(f"command failed (exit {proc.returncode}): {cmd[0]}")


def _check_osmium() -> None:
    if shutil.which("osmium") is None:
        sys.exit(
            "osmium not found on PATH. Install via 'brew install osmium-tool' "
            "or your package manager."
        )


def _extract_layer(
    src: Path, layer: str, filters: list[str], out_dir: Path, force: bool
) -> None:
    out_path = out_dir / f"kazan-agg-{layer}.geojsonseq"
    if out_path.exists() and not force:
        size_mb = out_path.stat().st_size / 1024 / 1024
        print(f"{out_path} already exists ({size_mb:.1f} MB); --force to rebuild")
        return

    pbf_path = out_dir / f"kazan-agg-{layer}.osm.pbf"
    _run(["osmium", "tags-filter", "--overwrite", "-o", str(pbf_path), str(src), *filters])
    _run(
        [
            "osmium",
            "export",
            "--overwrite",
            "-f",
            "geojsonseq",
            "--add-unique-id=type_id",
            "-o",
            str(out_path),
            str(pbf_path),
        ]
    )
    pbf_path.unlink(missing_ok=True)
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"Wrote {out_path} ({size_mb:.1f} MB)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--src", type=Path, default=_DEFAULT_SRC, help="Source PBF (clipped to agglomeration)"
    )
    p.add_argument(
        "--out-dir", type=Path, default=_OUT_DIR, help="Output directory"
    )
    p.add_argument(
        "--layers",
        nargs="+",
        default=list(_LAYER_FILTERS.keys()),
        choices=list(_LAYER_FILTERS.keys()),
        help="Layer names to extract (default: all)",
    )
    p.add_argument("--force", action="store_true", help="Rebuild even if output exists")
    args = p.parse_args()

    _check_osmium()
    if not args.src.is_file():
        sys.exit(f"source PBF does not exist: {args.src}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for layer in args.layers:
        _extract_layer(args.src, layer, _LAYER_FILTERS[layer], args.out_dir, args.force)


if __name__ == "__main__":
    main()
