"""Augment OSM water layer with ESA WorldCover 2021 water polygons.

Why
---
Even after the smart-extract pipeline fix (relations like Куйбышевское
водохранилище are now whole), OSM still has localised polygon gaps in
places mappers haven't traced — most visibly the Бишбалта peninsula in
the Казанка mouth, where waterway-axes are present but no closed water
polygon exists. Apartments there get ``dist_to_water_m`` ~1.2 km and
``water_share_500m`` = 0 even though they sit on the riverbank.

ESA WorldCover v200 (2021) is a global 10 m land-cover product derived
from Sentinel-1+2 by ESA, with a hard "water" class (id=80). At 10 m
it sees ports, harbours and narrow shore inlets that JRC GSW (30 m
occurrence) loses, and it does not depend on community mapping.

Output
------
- ``data/raw/esa_worldcover/ESA_WorldCover_10m_2021_v200_N54E048_Map.tif``
  cached download (~62 MB, covers 54-57°N × 48-51°E — Tatarstan).
- ``data/raw/osm/kazan-agg-water-esa.geojsonseq`` — ESA water polygons
  clipped to the agglomeration bbox.
- ``data/raw/osm/kazan-agg-water.geojsonseq`` — overwritten with the
  union of OSM smart-extract + ESA. The feature pipeline reads this
  one path so no config wiring needed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx
import numpy as np
import rasterio
from rasterio.features import shapes as raster_shapes
from rasterio.windows import from_bounds
from shapely.geometry import box, mapping, shape
from shapely.ops import transform as shapely_transform
from pyproj import Transformer

# ----------------------------------------------------------------------
# Single 3°×3° ESA tile covers the Kazan agglomeration in full.
# Tile naming: ``N{lat}E{lon}`` = SW corner. N54E048 → 54-57°N × 48-51°E.
_ESA_BASE_URL = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"
_ESA_TILE = "ESA_WorldCover_10m_2021_v200_N54E048_Map.tif"

# ESA WorldCover class encoding. We want only "water bodies" here —
# 90 ("herbaceous wetland") is excluded because flood-prone meadows are
# not what ``dist_to_water_m`` should measure.
_WATER_CLASS = 80

# Agglomeration bbox — wider than the OSM smart-extract bbox so ESA
# polygons aren't truncated mid-water at the edges.
_AGG_BBOX = (48.4, 55.3, 50.0, 56.1)

# ESA pixels are 10×10 m = ~100 m² in UTM. Threshold 50 m² keeps any
# single isolated water-class pixel because some legitimate features —
# narrow port slips, harbour basins, the inland tip of the Бишбалта
# inlet — show up as 1-2 pixel clusters that an aggressive filter wipes
# along with the noise. Empirically 50 m² catches Бишбалта (otherwise
# 1.2 km error remains) without flooding the layer with random pixels.
_MIN_POLYGON_AREA_M2 = 50.0


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


def _polygonize_water(
    raster_path: Path, bbox: tuple[float, float, float, float]
) -> list:
    with rasterio.open(raster_path) as src:
        if src.crs is None or src.crs.to_epsg() != 4326:
            sys.exit(f"unexpected CRS {src.crs}; expected EPSG:4326")

        window = from_bounds(*bbox, transform=src.transform)
        data = src.read(1, window=window)
        win_transform = src.window_transform(window)

        mask = data == _WATER_CLASS
        binary = mask.astype(np.uint8)

        polygons: list = []
        for geom_dict, value in raster_shapes(
            binary, mask=binary.astype(bool), transform=win_transform
        ):
            if value != 1:
                continue
            polygons.append(shape(geom_dict))

    return polygons


def _filter_and_clip(
    polygons: list, bbox: tuple[float, float, float, float], min_area_m2: float
) -> list:
    clip_box = box(*bbox)
    kept = []
    for poly in polygons:
        clipped = poly.intersection(clip_box)
        if clipped.is_empty:
            continue
        for part in (
            list(clipped.geoms)
            if clipped.geom_type in ("MultiPolygon", "GeometryCollection")
            else [clipped]
        ):
            if part.geom_type != "Polygon":
                continue
            area_m2 = _project_lonlat_to_utm(part).area
            if area_m2 < min_area_m2:
                continue
            kept.append(part)
    return kept


def _write_geojsonseq(polygons: list, out_path: Path, source_tag: str) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        for i, poly in enumerate(polygons):
            feat = {
                "type": "Feature",
                "id": f"esa/{source_tag}/{i}",
                "properties": {
                    "natural": "water",
                    "source": "esa_worldcover_v200_2021",
                },
                "geometry": mapping(poly),
            }
            f.write(b"\x1e")
            f.write(json.dumps(feat, ensure_ascii=False).encode("utf-8"))
            f.write(b"\n")
    return len(polygons)


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


def _augment_osm(
    osm_path: Path, esa_polygons: list, out_path: Path
) -> tuple[int, int]:
    """OSM smart-extract ∪ ESA. The downstream geom-distance pipeline does
    ``unary_union`` of all input geometries before building its STRtree,
    so overlapping polygons collapse naturally — we don't dedupe here.

    Idempotent re-runs: any pre-existing ESA features (from a prior run
    of this script) are stripped before re-writing, so running twice
    doesn't double-count."""
    osm_features = [
        f for f in _read_geojsonseq(osm_path)
        if not str(f.get("properties", {}).get("source", "")).startswith("esa_")
    ]
    n_esa_written = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        for feat in osm_features:
            f.write(b"\x1e")
            f.write(json.dumps(feat, ensure_ascii=False).encode("utf-8"))
            f.write(b"\n")
        for i, poly in enumerate(esa_polygons):
            feat = {
                "type": "Feature",
                "id": f"esa/water/{i}",
                "properties": {
                    "natural": "water",
                    "source": "esa_worldcover_v200_2021",
                },
                "geometry": mapping(poly),
            }
            f.write(b"\x1e")
            f.write(json.dumps(feat, ensure_ascii=False).encode("utf-8"))
            f.write(b"\n")
            n_esa_written += 1
    return len(osm_features), n_esa_written


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=Path("data/raw/osm"))
    p.add_argument("--cache-dir", type=Path, default=Path("data/raw/esa_worldcover"))
    p.add_argument(
        "--osm-water",
        type=Path,
        default=Path("data/raw/osm/kazan-agg-water.geojsonseq"),
        help="OSM water layer (overwritten with the OSM∪ESA union)",
    )
    p.add_argument(
        "--min-area",
        type=float,
        default=_MIN_POLYGON_AREA_M2,
        help="Drop ESA polygons smaller than this (m²)",
    )
    p.add_argument("--force", action="store_true", help="Re-download tile")
    args = p.parse_args()

    tile_path = args.cache_dir / _ESA_TILE
    _download_tile(tile_path, force=args.force)

    print(f"Polygonising ESA water class={_WATER_CLASS} (bbox={_AGG_BBOX}) …")
    polygons = _polygonize_water(tile_path, _AGG_BBOX)
    print(f"  raw water polygons: {len(polygons)}")

    polygons = _filter_and_clip(polygons, _AGG_BBOX, args.min_area)
    print(f"  after clip + min-area filter: {len(polygons)}")

    esa_path = args.out_dir / "kazan-agg-water-esa.geojsonseq"
    n = _write_geojsonseq(polygons, esa_path, "water")
    size_mb = esa_path.stat().st_size / 1024 / 1024
    print(f"Wrote {esa_path} ({size_mb:.2f} MB, {n} ESA polygons)")

    n_osm, n_esa = _augment_osm(args.osm_water, polygons, args.osm_water)
    size_mb = args.osm_water.stat().st_size / 1024 / 1024
    print(
        f"Overwrote {args.osm_water} ({size_mb:.2f} MB, "
        f"{n_osm} OSM + {n_esa} ESA = {n_osm + n_esa} features)"
    )


if __name__ == "__main__":
    main()
