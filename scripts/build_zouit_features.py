"""Build the silver per-object ЗОУИТ feature table (ADR-0025, п. 3, stage 2).

Spatial join of valuation-object points against the silver zone layer
built by ``scripts/build_zouit_silver.py`` (point-in-polygon via
STRtree over UTM-39N projected geometries).

Output:

    data/silver/zouit_per_object/region={code}/data.parquet
      object_id, inside_zouit, zouit_types, inside_water_protection

Запуск:
    uv run python scripts/build_zouit_features.py
"""

from __future__ import annotations

import sys

import polars as pl

from kadastra.adapters.parquet_valuation_object_store import ParquetValuationObjectStore
from kadastra.config import Settings
from kadastra.domain.asset_class import AssetClass
from kadastra.etl.object_zouit_features import compute_object_zouit_features


def main() -> int:
    settings = Settings()

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

    zones_path = settings.zouit_silver_path / f"region={settings.region_code}" / "data.parquet"
    if not zones_path.is_file():
        sys.exit(f"zone layer missing: {zones_path} — run build_zouit_silver.py first")
    zones = pl.read_parquet(zones_path)
    print(f"=> zones: {zones.height:,}", flush=True)

    features = compute_object_zouit_features(objects, zones=zones)

    out_dir = settings.zouit_features_path / f"region={settings.region_code}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "data.parquet"
    features.write_parquet(out_path)
    print(f"=> wrote {out_path}  rows={features.height:,}", flush=True)

    print("\n=> feature coverage (non-null share):")
    for col in ("inside_zouit", "zouit_types", "inside_water_protection"):
        share = features.select(pl.col(col).is_not_null().mean()).item()
        print(f"   {col}: {share:.1%}", flush=True)
    inside = features.filter(pl.col("inside_zouit") == 1)
    print(f"\n=> objects inside any ЗОУИТ: {inside.height:,} ({inside.height / features.height:.1%})")
    in_water = features.filter(pl.col("inside_water_protection") == 1)
    print(f"=> objects inside water_protection: {in_water.height:,} ({in_water.height / features.height:.1%})")
    print("\n=> top zouit_types combinations:")
    for row in inside.group_by("zouit_types").len().sort("len", descending=True).head(15).iter_rows():
        print(f"   {row[0]}: {row[1]:,}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
