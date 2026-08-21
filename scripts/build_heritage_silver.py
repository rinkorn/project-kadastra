"""Build the silver ОКН (cultural heritage) layer (ADR-0025, п. 2).

Source: the OSM extract ``Settings.heritage_raw_geojson_path``
(``kazan-agg-heritage.geojsonseq``) — the Минкульт open-data API
(``opendata.mkrf.ru``) is unreachable from our network (connection
refused), so OSM ``heritage=*`` objects are the substitute source
(S3 backup: ``Kadatastr/raw/osm/kazan-agg-heritage.geojsonseq``).

The extract carries service/non-ОКН rows (entrances, gates, barriers) —
``parse_heritage_geojsonseq`` keeps only features with a non-null
``heritage`` tag.

Output:

    data/silver/heritage/region={code}/data.parquet
      osm_id, ref_egrokn, heritage_level, name, lat, lon, polygon_wkt

Запуск:
    uv run python scripts/build_heritage_silver.py
"""

from __future__ import annotations

import sys

from kadastra.config import Settings
from kadastra.etl.object_heritage_features import parse_heritage_geojsonseq


def main() -> int:
    settings = Settings()

    raw_path = settings.heritage_raw_geojson_path
    if not raw_path.is_file():
        sys.exit(f"raw heritage extract missing: {raw_path}")

    with raw_path.open("r", encoding="utf-8") as f:
        heritage = parse_heritage_geojsonseq(f)
    print(f"=> parsed ОКН rows: {heritage.height:,}", flush=True)
    if heritage.height:
        with_polygon = heritage.filter(heritage["polygon_wkt"].is_not_null()).height
        with_ref = heritage.filter(heritage["ref_egrokn"].is_not_null()).height
        print(f"=> with polygon footprint: {with_polygon:,}", flush=True)
        print(f"=> with ref:egrokn: {with_ref:,}", flush=True)

    out_dir = settings.heritage_silver_path / f"region={settings.region_code}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "data.parquet"
    heritage.write_parquet(out_path)
    print(f"=> wrote {out_path}  rows={heritage.height:,}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
