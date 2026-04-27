"""Extract OSM layers (polygonal, point-POI and linear) from the
full Volga federal district PBF into per-layer GeoJSON-seq files.

Consumed by:
- ``compute_object_polygon_features`` (poly-area buffer share — ADR-0014);
- ``compute_object_geom_distance_features`` (distance to nearest
  geometry of any type — ADR-0019);
- ``compute_object_municipality_features`` (raion spatial join — ADR-0015);
- ``compute_object_zonal_features`` (POI counts in radii — ADR-0013).

Source: ``data/raw/osm/volga-fed-district-latest.osm.pbf`` (~720 MB
Geofabrik download). We deliberately do NOT use the pre-clipped
``kazan-agg.osm.pbf`` because that file was made with a non-smart bbox
clip — multi-region relations (e.g. ``Куйбышевское водохранилище`` whose
member ways span 5+ regions) lose ways, and ``osmium export`` then
silently drops the unclosed polygon. Result: huge water bodies just
disappear from the water layer, and ``dist_to_water_m`` /
``water_share_500m`` lie about reality.

The fix is to ``osmium extract --strategy=smart`` AFTER ``tags-filter``,
which preserves complete relations whose envelope merely intersects the
bbox — Куйбышевское-relation comes through whole, polygon assembles
correctly.

Outputs (one per layer): ``data/raw/osm/kazan-agg-{layer}.geojsonseq``

For point POIs that can be mapped as either nodes or polygons (school,
hospital, supermarket), the filter uses ``nwa/`` so both forms land in
the same file; downstream code centroids polygon features when a Point
is needed (zonal counts) and uses geometries as-is for distance.

The script is idempotent: skips any layer whose output already exists,
unless ``--force`` is passed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# OSM tag filter expressions, one per layer.
# Format follows osmium-tool's `tags-filter` rules: `<object-types>/<key=value>`.
# Object-type prefixes: `n` nodes, `w` ways, `r` relations, `a` areas
# (closed way OR multipolygon relation). `nwa` = node + way + area.
_LAYER_FILTERS: dict[str, list[str]] = {
    # — Polygonal: poly-area share (ADR-0014) + distance (ADR-0019). —
    "water": [
        "wa/natural=water",
        "wa/waterway=riverbank",
        "wa/landuse=reservoir",
    ],
    "park": [
        "wa/leisure=park",
        "wa/leisure=garden",
    ],
    "forest": [
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
    "landfill": [
        "wa/landuse=landfill",
    ],
    # — Linear: distance to nearest line (ADR-0019). —
    # `w/` only: power lines and rails are linear ways, not areas.
    "powerline": [
        "w/power=line",
        "w/power=minor_line",
    ],
    "railway": [
        "w/railway=rail",
    ],
    # — Point-POI: distance + zonal counts (ADR-0019). —
    # Use `nwa/` for amenities that can be mapped as either a node or
    # a polygon (large schools/hospitals are usually areas; small
    # clinics/cafes are usually nodes). Downstream centroids polygons
    # for zonal counts and uses geometries as-is for distance.
    "school": ["nwa/amenity=school"],
    "kindergarten": ["nwa/amenity=kindergarten"],
    "clinic": [
        "nwa/amenity=clinic",
        "nwa/amenity=doctors",
    ],
    "hospital": ["nwa/amenity=hospital"],
    "pharmacy": ["nwa/amenity=pharmacy"],
    "supermarket": [
        "nwa/shop=supermarket",
        "nwa/shop=mall",
    ],
    "cafe": ["nwa/amenity=cafe"],
    "restaurant": ["nwa/amenity=restaurant"],
    # bus stops are conventionally nodes (`highway=bus_stop`); rail
    # stops/stations can be either node (the platform anchor) or way
    # (the platform polyline) — use `nwa/` defensively.
    "bus_stop": ["n/highway=bus_stop"],
    "tram_stop": ["nwa/railway=tram_stop"],
    "railway_station": ["nwa/railway=station"],
}

# Raions are chained-filter: only relations with both
# boundary=administrative AND admin_level=9. osmium tags-filter combines
# expressions with OR within one invocation, so we run two passes.
# Output is then post-filtered to keep only Polygon/MultiPolygon
# features whose properties carry admin_level=9 (osmium export emits
# referenced nodes/ways too — they get tossed here).
_RAIONS_FILTER_PASSES: list[list[str]] = [
    ["r/boundary=administrative"],
    ["r/admin_level=9"],
]

_DEFAULT_SRC = Path("data/raw/osm/volga-fed-district-latest.osm.pbf")
_OUT_DIR = Path("data/raw/osm")

# Bbox of the Kazan agglomeration — same envelope previously baked into
# ``kazan-agg.osm.pbf`` so downstream features (raion lookup, share-in-
# buffer, distance) see the same region. Format: minlon,minlat,maxlon,maxlat.
_DEFAULT_BBOX = "48.4,55.3,50.0,56.1"


def _run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        sys.exit(f"command failed (exit {proc.returncode}): {cmd[0]}")


def _check_osmium() -> None:
    if shutil.which("osmium") is None:
        sys.exit("osmium not found on PATH. Install via 'brew install osmium-tool' or your package manager.")


def _extract_layer(
    src: Path,
    layer: str,
    filters: list[str],
    out_dir: Path,
    bbox: str,
    force: bool,
) -> None:
    out_path = out_dir / f"kazan-agg-{layer}.geojsonseq"
    if out_path.exists() and not force:
        size_mb = out_path.stat().st_size / 1024 / 1024
        print(f"{out_path} already exists ({size_mb:.1f} MB); --force to rebuild")
        return

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        # Stage 1: tag filter on the full PBF — narrows ~720 MB → tens of
        # MB by keeping only the layer's tagged objects, but at this point
        # multi-region relations (e.g. Куйбышевское water relation) still
        # carry all their member ways, even those well outside the bbox.
        filtered = tmp_path / f"{layer}-tagged.osm.pbf"
        _run(
            [
                "osmium",
                "tags-filter",
                "--overwrite",
                "-o",
                str(filtered),
                str(src),
                *filters,
            ]
        )
        # Stage 2: smart bbox clip — keeps ways whose nodes are inside the
        # bbox AND complete relations whose envelope merely touches it.
        # Without --strategy=smart, the relation's ways outside the bbox
        # are dropped → broken rings → polygon silently disappears.
        clipped = tmp_path / f"{layer}-clipped.osm.pbf"
        _run(
            [
                "osmium",
                "extract",
                "--overwrite",
                "--strategy=smart",
                "-b",
                bbox,
                "-o",
                str(clipped),
                str(filtered),
            ]
        )
        # Stage 3: vector export — osmium assembles closed polygons from
        # complete relations now that no ways are missing.
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
                str(clipped),
            ]
        )

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"Wrote {out_path} ({size_mb:.1f} MB)")


def _extract_raions(src: Path, out_dir: Path, bbox: str, force: bool) -> None:
    out_path = out_dir / "kazan-agg-raions.geojsonseq"
    if out_path.exists() and not force:
        size_mb = out_path.stat().st_size / 1024 / 1024
        print(f"{out_path} already exists ({size_mb:.1f} MB); --force to rebuild")
        return

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        current = src
        for i, filters in enumerate(_RAIONS_FILTER_PASSES):
            stage = tmp_path / f"raions-stage{i}.osm.pbf"
            _run(["osmium", "tags-filter", "--overwrite", "-o", str(stage), str(current), *filters])
            current = stage
        # Smart bbox clip keeps complete admin relations (raion polygons
        # are made of many way members; truncating any of them produces
        # null polygons just like the water-layer Куйбышевское bug).
        clipped = tmp_path / "raions-clipped.osm.pbf"
        _run(
            [
                "osmium",
                "extract",
                "--overwrite",
                "--strategy=smart",
                "-b",
                bbox,
                "-o",
                str(clipped),
                str(current),
            ]
        )
        current = clipped
        raw = tmp_path / "raions-raw.geojsonseq"
        _run(
            [
                "osmium",
                "export",
                "--overwrite",
                "-f",
                "geojsonseq",
                "--add-unique-id=type_id",
                "-o",
                str(raw),
                str(current),
            ]
        )
        kept = 0
        with raw.open("rb") as fin, out_path.open("wb") as fout:
            for chunk in fin.read().split(b"\x1e"):
                line = chunk.strip()
                if not line:
                    continue
                try:
                    feat = json.loads(line)
                except json.JSONDecodeError:
                    continue
                geom_type = (feat.get("geometry") or {}).get("type")
                props = feat.get("properties") or {}
                if geom_type not in ("Polygon", "MultiPolygon"):
                    continue
                if str(props.get("admin_level")) != "9":
                    continue
                fout.write(b"\x1e")
                fout.write(json.dumps(feat, ensure_ascii=False).encode("utf-8"))
                fout.write(b"\n")
                kept += 1
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"Wrote {out_path} ({size_mb:.1f} MB, {kept} admin polygons)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    all_layers = [*list(_LAYER_FILTERS.keys()), "raions"]
    p.add_argument(
        "--src",
        type=Path,
        default=_DEFAULT_SRC,
        help="Full Volga federal district PBF (extract clips with --strategy=smart)",
    )
    p.add_argument(
        "--bbox",
        type=str,
        default=_DEFAULT_BBOX,
        help="Agglomeration bbox (minlon,minlat,maxlon,maxlat) for the smart clip",
    )
    p.add_argument("--out-dir", type=Path, default=_OUT_DIR, help="Output directory")
    p.add_argument(
        "--layers",
        nargs="+",
        default=all_layers,
        choices=all_layers,
        help="Layer names to extract (default: all)",
    )
    p.add_argument("--force", action="store_true", help="Rebuild even if output exists")
    args = p.parse_args()

    _check_osmium()
    if not args.src.is_file():
        sys.exit(f"source PBF does not exist: {args.src}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for layer in args.layers:
        if layer == "raions":
            _extract_raions(args.src, args.out_dir, args.bbox, args.force)
        else:
            _extract_layer(
                args.src,
                layer,
                _LAYER_FILTERS[layer],
                args.out_dir,
                args.bbox,
                args.force,
            )


if __name__ == "__main__":
    main()
