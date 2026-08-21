"""Build the silver ЗОУИТ zone layer (ADR-0025, п. 3, stage 1).

Source: bulk dump of НСПД layer 36302 (ЗОУИТ) for Tatarstan —
``Settings.zouit_raw_dir`` (``page-*.json``, 177 663 objects,
EPSG:3857 geometries, ``properties.options.type_zone`` kind strings).
The ADR's hypothetical ``attrs.zouit_intersection`` field does not
exist; the zone polygons are the real source (S3 backup:
``Kadatastr/raw/nspd/zouit-tatarstan/``).

The dump covers all of Tatarstan, so zones are pre-filtered to the
region boundary bbox (expanded by ``_BBOX_MARGIN_M``) — this cuts the
layer by orders of magnitude before the per-object spatial join.

Output:

    data/silver/zouit_zones/region={code}/data.parquet
      zouit_id, type_zone, category, geometry_wkt_3857

Запуск:
    uv run python scripts/build_zouit_silver.py
"""

from __future__ import annotations

import json
import sys
from typing import Any

import polars as pl
from pyproj import Transformer
from shapely.ops import transform as shapely_transform

from kadastra.adapters.local_geojson_region_boundary import LocalGeoJsonRegionBoundary
from kadastra.config import Settings
from kadastra.etl.object_zouit_features import (
    ZOUIT_ZONE_SCHEMA,
    parse_zouit_feature,
)

# Margin around the region bbox, in EPSG:3857 metres. Zones touching
# the boundary may still cover objects inside, so the bbox is padded.
_BBOX_MARGIN_M = 10_000.0

_TO_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


def _region_bbox_3857(settings: Settings) -> tuple[float, float, float, float]:
    boundary = LocalGeoJsonRegionBoundary(
        settings.region_boundary_path,
        settings.region_boundary_field,
    ).get_boundary(settings.region_code)
    projected = shapely_transform(lambda x, y, z=None: _TO_3857.transform(x, y), boundary)
    minx, miny, maxx, maxy = projected.bounds
    return (
        minx - _BBOX_MARGIN_M,
        miny - _BBOX_MARGIN_M,
        maxx + _BBOX_MARGIN_M,
        maxy + _BBOX_MARGIN_M,
    )


def _coords_bbox(coords: Any) -> tuple[float, float, float, float] | None:
    """Min/max over a nested GeoJSON coordinate structure."""
    minx = miny = float("inf")
    maxx = maxy = float("-inf")
    stack = [coords]
    while stack:
        item = stack.pop()
        if isinstance(item, (list, tuple)):
            if len(item) >= 2 and isinstance(item[0], (int, float)):
                x, y = float(item[0]), float(item[1])
                minx, miny = min(minx, x), min(miny, y)
                maxx, maxy = max(maxx, x), max(maxy, y)
            else:
                stack.extend(item)
    if minx > maxx:
        return None
    return minx, miny, maxx, maxy


def _intersects(bbox: tuple[float, float, float, float], region: tuple[float, float, float, float]) -> bool:
    minx, miny, maxx, maxy = bbox
    rminx, rminy, rmaxx, rmaxy = region
    return minx <= rmaxx and maxx >= rminx and miny <= rmaxy and maxy >= rminy


def main() -> int:
    settings = Settings()

    raw_dir = settings.zouit_raw_dir
    pages = sorted(raw_dir.glob("page-*.json"))
    if not pages:
        sys.exit(f"no ЗОУИТ page dump found in {raw_dir}")
    print(f"=> pages: {len(pages)}", flush=True)

    region_bbox = _region_bbox_3857(settings)
    print(f"=> region bbox (3857, +{_BBOX_MARGIN_M:.0f} m margin): {region_bbox}", flush=True)

    rows: list[dict[str, Any]] = []
    total = 0
    for page_path in pages:
        with page_path.open("r", encoding="utf-8") as f:
            page = json.load(f)
        for feature in (page.get("data") or {}).get("features") or []:
            total += 1
            geom = feature.get("geometry")
            if geom is None:
                continue
            bbox = _coords_bbox(geom.get("coordinates"))
            if bbox is None or not _intersects(bbox, region_bbox):
                continue
            row = parse_zouit_feature(feature)
            if row is not None:
                rows.append(row)
        print(f"=> {page_path.name}: total={total:,} kept={len(rows):,}", flush=True)

    zones = pl.DataFrame(rows, schema=ZOUIT_ZONE_SCHEMA)
    out_dir = settings.zouit_silver_path / f"region={settings.region_code}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "data.parquet"
    zones.write_parquet(out_path)
    print(f"=> wrote {out_path}  rows={zones.height:,} (of {total:,} dumped)", flush=True)

    print("\n=> category distribution (kept zones):")
    for row in zones.group_by("category").len().sort("len", descending=True).iter_rows():
        print(f"   {row[0]}: {row[1]:,}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
