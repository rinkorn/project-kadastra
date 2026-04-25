"""Convert an osmium-exported GeoJSON-seq of building polygons into the CSV
schema that BuildBuildingsFeatures already consumes:

    osm_id,osm_type,lat,lon,building,levels,flats,material,wall,street,
    housenumber,postcode,city,start_date,name,energy_class

Centroids are computed in WGS84 (good enough for hex assignment at res=10).
Non-building features (e.g. entrance Points) are skipped.
"""

import csv
import json
import sys
from pathlib import Path

from shapely.geometry import shape

INPUT_PATH = Path("data/raw/osm/kazan-agg-buildings.geojsonseq")
OUTPUT_PATH = Path("data/raw/osm/osm_buildings_kazan_agglomeration.csv")

CSV_COLUMNS = [
    "osm_id",
    "osm_type",
    "lat",
    "lon",
    "building",
    "levels",
    "flats",
    "material",
    "wall",
    "street",
    "housenumber",
    "postcode",
    "city",
    "start_date",
    "name",
    "energy_class",
]

PROPERTY_MAP = {
    "building": "building",
    "building:levels": "levels",
    "building:flats": "flats",
    "building:material": "material",
    "wall": "wall",
    "addr:street": "street",
    "addr:housenumber": "housenumber",
    "addr:postcode": "postcode",
    "addr:city": "city",
    "start_date": "start_date",
    "name": "name",
    "building:energy_class": "energy_class",
}


_TYPE_PREFIX_MAP = {
    "n": "node",
    "w": "way",
    "r": "relation",
    "a": "way",  # osmium 'area' IDs are most often closed ways promoted to areas
}


def _parse_id(raw_id: str | None) -> tuple[str, str]:
    """Parses osmium's --add-unique-id=type_id format: 'a103502712' → ('way', '103502712')."""
    if not raw_id:
        return "", ""
    if "/" in raw_id:
        osm_type, osm_id = raw_id.split("/", 1)
        return osm_type, osm_id
    if raw_id and raw_id[0] in _TYPE_PREFIX_MAP and raw_id[1:].isdigit():
        return _TYPE_PREFIX_MAP[raw_id[0]], raw_id[1:]
    return "", raw_id


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0

    with INPUT_PATH.open("rb") as fin, OUTPUT_PATH.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()

        for raw in fin:
            line = raw.lstrip(b"\x1e").strip()
            if not line:
                continue
            try:
                feat = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            geom_type = feat.get("geometry", {}).get("type")
            props = feat.get("properties", {}) or {}
            if geom_type not in ("Polygon", "MultiPolygon") or not props.get("building"):
                skipped += 1
                continue

            try:
                centroid = shape(feat["geometry"]).centroid
            except Exception:
                skipped += 1
                continue

            osm_type, osm_id = _parse_id(feat.get("id"))
            row: dict[str, object] = {
                "osm_id": osm_id,
                "osm_type": osm_type,
                "lat": round(centroid.y, 7),
                "lon": round(centroid.x, 7),
            }
            for src_key, dst_key in PROPERTY_MAP.items():
                if src_key in props:
                    row[dst_key] = props[src_key]
            writer.writerow(row)
            written += 1

    print(f"Wrote {written} buildings to {OUTPUT_PATH} (skipped {skipped} non-building features)")


if __name__ == "__main__":
    main()
    sys.exit(0)
