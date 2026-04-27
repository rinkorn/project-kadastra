"""Augment an OSM polygonal layer with ESA WorldCover 10 m raster polygons.

Why
---
OSM is community-mapped. Even after the smart-extract pipeline fix
(cross-region relations like Куйбышевское водохранилище assemble whole),
locally many polygons are simply missing — mappers traced waterway-axes
without closing them at the Казанка mouth, traced individual park
polygons but skipped neighbouring forest patches, etc. The result:
``dist_to_<layer>_m`` overshoots and ``<layer>_share_500m`` underflows
for the buildings/landplots near those gaps.

ESA WorldCover v200 (2021) is a 10 m global land-cover product derived
from Sentinel-1+2 with hard class IDs (80=water, 10=tree cover, ...).
At 10 m it sees ports, harbours, narrow forest patches that OSM misses
and is independent of community mapping. We polygonize the relevant
class, clip to the agglomeration bbox, and merge into the OSM layer
file consumed by the feature pipeline. The merge strips any prior ESA
features first, so re-runs are idempotent.

Usage
-----
``python scripts/build_esa_landcover_layer.py`` (defaults: class 80 = water)
augments ``data/raw/osm/kazan-agg-water.geojsonseq``.

``python scripts/build_esa_landcover_layer.py --class-id 10 --layer forest``
augments ``data/raw/osm/kazan-agg-forest.geojsonseq`` with tree cover
that OSM ``landuse=forest`` mappers haven't traced yet.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.features import shapes as raster_shapes
from rasterio.windows import from_bounds
from shapely.geometry import box, mapping, shape
from shapely.ops import transform as shapely_transform

# Single 3°×3° ESA tile covers the Kazan agglomeration in full.
# Tile naming: ``N{lat}E{lon}`` = SW corner. N54E048 → 54-57°N × 48-51°E.
_ESA_BASE_URL = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"
_ESA_TILE = "ESA_WorldCover_10m_2021_v200_N54E048_Map.tif"

# ESA WorldCover class encoding (subset of interest).
_KNOWN_CLASSES: dict[int, str] = {
    10: "tree_cover",
    20: "shrubland",
    30: "grassland",
    50: "built_up",
    80: "water",
    90: "wetland",
}

# Agglomeration bbox — wider than the OSM smart-extract bbox so ESA
# polygons aren't truncated mid-feature at the edges.
_AGG_BBOX = (48.4, 55.3, 50.0, 56.1)

# ESA pixels are 10×10 m = ~100 m² in UTM. Threshold 50 m² keeps any
# single isolated class pixel because some legitimate features —
# narrow port slips, harbour basins, isolated forest patches — show
# up as 1-2 pixel clusters that an aggressive filter wipes along
# with the noise. Empirically this catches Бишбалта (1148 → 473 m
# on water) without flooding the layer with random pixels.
_DEFAULT_MIN_AREA_M2 = 50.0


_TO_UTM = Transformer.from_crs("EPSG:4326", "EPSG:32639", always_xy=True)


def _project_lonlat_to_utm(geom):
    return shapely_transform(lambda x, y, z=None: _TO_UTM.transform(x, y), geom)


def _download_tile(out_path: Path, force: bool = False) -> None:
    if out_path.exists() and not force:
        size_mb = out_path.stat().st_size / 1024 / 1024
        print(f"{out_path} already exists ({size_mb:.1f} MB); --force to redownload")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{_ESA_BASE_URL}/{_ESA_TILE}"
    print(f"Downloading {url} → {out_path}")
    with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with out_path.open("wb") as f:
            for chunk in resp.iter_bytes(chunk_size=1 << 20):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = 100 * downloaded / total
                    print(
                        f"  {downloaded / 1024 / 1024:.1f}/{total / 1024 / 1024:.1f} MB ({pct:.0f}%)",
                        end="\r",
                    )
        print()


def _polygonize_class(
    raster_path: Path,
    class_id: int,
    bbox: tuple[float, float, float, float],
) -> list:
    with rasterio.open(raster_path) as src:
        if src.crs is None or src.crs.to_epsg() != 4326:
            sys.exit(f"unexpected CRS {src.crs}; expected EPSG:4326")

        window = from_bounds(*bbox, transform=src.transform)
        data = src.read(1, window=window)
        win_transform = src.window_transform(window)

        mask = data == class_id
        binary = mask.astype(np.uint8)

        polygons: list = []
        for geom_dict, value in raster_shapes(binary, mask=binary.astype(bool), transform=win_transform):
            if value != 1:
                continue
            polygons.append(shape(geom_dict))

    return polygons


def _filter_and_clip(polygons: list, bbox: tuple[float, float, float, float], min_area_m2: float) -> list:
    clip_box = box(*bbox)
    kept = []
    for poly in polygons:
        clipped = poly.intersection(clip_box)
        if clipped.is_empty:
            continue
        for part in list(clipped.geoms) if clipped.geom_type in ("MultiPolygon", "GeometryCollection") else [clipped]:
            if part.geom_type != "Polygon":
                continue
            area_m2 = _project_lonlat_to_utm(part).area
            if area_m2 < min_area_m2:
                continue
            kept.append(part)
    return kept


def _osm_property_for_layer(layer: str) -> dict[str, str]:
    """OSM-equivalent tags on the synthetic ESA features so downstream
    filters that key off ``natural=water`` / ``landuse=forest`` see them
    just like real OSM polygons. The ``source`` field is what the
    idempotent re-augment step uses to distinguish ESA from native."""
    if layer == "water":
        return {"natural": "water"}
    if layer == "forest":
        return {"landuse": "forest"}
    return {layer: "yes"}


def _read_geojsonseq(path: Path) -> list:
    out = []
    with path.open("rb") as f:
        for rec in f.read().split(b"\x1e"):
            rec = rec.strip()
            if not rec:
                continue
            try:
                feat = json.loads(rec)
            except json.JSONDecodeError:
                continue
            out.append(feat)
    return out


def _write_geojsonseq(polygons: list, out_path: Path, layer: str) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base_props = {
        **_osm_property_for_layer(layer),
        "source": "esa_worldcover_v200_2021",
    }
    with out_path.open("wb") as f:
        for i, poly in enumerate(polygons):
            feat = {
                "type": "Feature",
                "id": f"esa/{layer}/{i}",
                "properties": dict(base_props),
                "geometry": mapping(poly),
            }
            f.write(b"\x1e")
            f.write(json.dumps(feat, ensure_ascii=False).encode("utf-8"))
            f.write(b"\n")
    return len(polygons)


def _augment_osm(osm_path: Path, esa_polygons: list, layer: str) -> tuple[int, int]:
    """OSM smart-extract ∪ ESA. The downstream geom-distance pipeline does
    ``unary_union`` of all input geometries before building the STRtree,
    so overlapping polygons collapse naturally — no spatial dedup here.

    Idempotent re-runs: any pre-existing ESA features (from a prior run
    of this script on the same layer) are stripped before re-writing,
    so running twice doesn't double-count."""
    osm_features = [
        f for f in _read_geojsonseq(osm_path) if not str(f.get("properties", {}).get("source", "")).startswith("esa_")
    ]
    base_props = {
        **_osm_property_for_layer(layer),
        "source": "esa_worldcover_v200_2021",
    }
    n_esa_written = 0
    osm_path.parent.mkdir(parents=True, exist_ok=True)
    with osm_path.open("wb") as f:
        for feat in osm_features:
            f.write(b"\x1e")
            f.write(json.dumps(feat, ensure_ascii=False).encode("utf-8"))
            f.write(b"\n")
        for i, poly in enumerate(esa_polygons):
            feat = {
                "type": "Feature",
                "id": f"esa/{layer}/{i}",
                "properties": dict(base_props),
                "geometry": mapping(poly),
            }
            f.write(b"\x1e")
            f.write(json.dumps(feat, ensure_ascii=False).encode("utf-8"))
            f.write(b"\n")
            n_esa_written += 1
    return len(osm_features), n_esa_written


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--class-id",
        type=int,
        default=80,
        help=f"ESA WorldCover class id. Common: {_KNOWN_CLASSES} (default 80=water)",
    )
    p.add_argument(
        "--layer",
        type=str,
        default="water",
        help="Layer name (matches kazan-agg-{layer}.geojsonseq); default 'water'",
    )
    p.add_argument("--out-dir", type=Path, default=Path("data/raw/osm"))
    p.add_argument("--cache-dir", type=Path, default=Path("data/raw/esa_worldcover"))
    p.add_argument(
        "--osm-path",
        type=Path,
        default=None,
        help="OSM layer path to augment (default: kazan-agg-{layer}.geojsonseq)",
    )
    p.add_argument(
        "--min-area",
        type=float,
        default=_DEFAULT_MIN_AREA_M2,
        help="Drop ESA polygons smaller than this (m²)",
    )
    p.add_argument("--force", action="store_true", help="Re-download tile")
    args = p.parse_args()

    osm_path = args.osm_path or args.out_dir / f"kazan-agg-{args.layer}.geojsonseq"

    tile_path = args.cache_dir / _ESA_TILE
    _download_tile(tile_path, force=args.force)

    print(f"Polygonising ESA class={args.class_id} ({_KNOWN_CLASSES.get(args.class_id, '?')}) bbox={_AGG_BBOX} …")
    polygons = _polygonize_class(tile_path, args.class_id, _AGG_BBOX)
    print(f"  raw class polygons: {len(polygons)}")

    polygons = _filter_and_clip(polygons, _AGG_BBOX, args.min_area)
    print(f"  after clip + min-area filter: {len(polygons)}")

    esa_only_path = args.out_dir / f"kazan-agg-{args.layer}-esa.geojsonseq"
    n = _write_geojsonseq(polygons, esa_only_path, args.layer)
    size_mb = esa_only_path.stat().st_size / 1024 / 1024
    print(f"Wrote {esa_only_path} ({size_mb:.2f} MB, {n} ESA polygons)")

    n_osm, n_esa = _augment_osm(osm_path, polygons, args.layer)
    size_mb = osm_path.stat().st_size / 1024 / 1024
    print(f"Overwrote {osm_path} ({size_mb:.2f} MB, {n_osm} OSM + {n_esa} ESA = {n_osm + n_esa} features)")


if __name__ == "__main__":
    main()
