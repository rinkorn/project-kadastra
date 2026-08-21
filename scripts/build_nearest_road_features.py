"""Build the silver ``road_class_per_object`` table (ADR-0024, group 1).

Sources:
- major road classes (motorway..tertiary + ``*_link``) — the Overpass
  raw ``Settings.roads_key`` (``tatarstan_major_roads.json``), read via
  S3 RawDataPort like the object pipeline does;
- minor road classes (residential/service/pedestrian/...) — the local
  parquet ``Settings.minor_road_ways_path``
  (``osm_id, highway, coords_json`` with ``[lon, lat]`` pairs). The
  existing road graph edges carry no highway classes (ADR-0024 «Аудит
  данных», п. 1), so both raws are merged here instead.

Output:

    data/silver/road_class_per_object/region={code}/data.parquet
      object_id, nearest_road_class, dist_to_{motorway,primary,
      secondary,residential,pedestrian}_m

Запуск:
    uv run python scripts/build_nearest_road_features.py
"""

from __future__ import annotations

import json
import sys
from typing import Any

import polars as pl

from kadastra.adapters.parquet_valuation_object_store import ParquetValuationObjectStore
from kadastra.composition_root import Container
from kadastra.config import Settings
from kadastra.domain.asset_class import AssetClass
from kadastra.etl.object_road_class_features import (
    compute_nearest_road_features,
    normalize_highway_class,
)


def load_major_ways(raw_payload: bytes) -> dict[str, list[list[tuple[float, float]]]]:
    """Parse the Overpass major-roads JSON into normalized ways_by_class."""
    payload: dict[str, Any] = json.loads(raw_payload)
    ways_by_class: dict[str, list[list[tuple[float, float]]]] = {}
    for element in payload.get("elements", []) or []:
        if element.get("type") != "way" or not element.get("geometry"):
            continue
        cls = normalize_highway_class((element.get("tags") or {}).get("highway"))
        if cls is None:
            continue
        way = [(float(p["lat"]), float(p["lon"])) for p in element["geometry"]]
        ways_by_class.setdefault(cls, []).append(way)
    return ways_by_class


def load_minor_ways(path) -> dict[str, list[list[tuple[float, float]]]]:
    """Parse the minor-roads parquet (coords_json = [[lon, lat], ...])."""
    ways_by_class: dict[str, list[list[tuple[float, float]]]] = {}
    df = pl.read_parquet(path)
    for highway, coords_json in df.select(["highway", "coords_json"]).iter_rows():
        cls = normalize_highway_class(highway)
        if cls is None:
            continue
        pairs = json.loads(coords_json)
        way = [(float(lat), float(lon)) for lon, lat in pairs]
        ways_by_class.setdefault(cls, []).append(way)
    return ways_by_class


def main() -> int:
    settings = Settings()
    container = Container(settings)

    store = ParquetValuationObjectStore(settings.valuation_object_store_path)
    frames = []
    for ac in AssetClass:
        df = store.load(settings.region_code, ac)
        if not df.is_empty():
            frames.append(df.select(["object_id", "lat", "lon"]))
    if not frames:
        sys.exit("no valuation objects found — run assemble/build pipeline first")
    objects = pl.concat(frames)
    print(f"=> objects: {objects.height:,}", flush=True)

    raw_data = container.build_s3_raw_data()
    ways_by_class = load_major_ways(raw_data.read_bytes(settings.roads_key))
    minor = load_minor_ways(settings.minor_road_ways_path)
    for cls, ways in minor.items():
        ways_by_class.setdefault(cls, []).extend(ways)
    print(
        "=> ways by class: " + ", ".join(f"{c}={len(w):,}" for c, w in sorted(ways_by_class.items())),
        flush=True,
    )

    features = compute_nearest_road_features(objects, ways_by_class=ways_by_class)

    out_dir = settings.road_class_features_path / f"region={settings.region_code}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "data.parquet"
    features.write_parquet(out_path)
    print(f"=> wrote {out_path}  rows={features.height:,}", flush=True)

    print("\n=> feature coverage (non-null share):")
    for col in features.columns:
        if col == "object_id":
            continue
        share = features.select(pl.col(col).is_not_null().mean()).item()
        print(f"   {col}: {share:.1%}", flush=True)
    print("\n=> nearest_road_class distribution:")
    for row in features.group_by("nearest_road_class").len().sort("len", descending=True).iter_rows():
        print(f"   {row[0]}: {row[1]:,}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
