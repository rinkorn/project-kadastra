"""Build a GeoJSON boundary for the Kazan agglomeration as a buffer around the
city center, projected to UTM zone 39N for accurate metric distances.

Output: data/raw/regions/kazan-agglomeration.geojson
- Single Feature with property `shapeISO=RU-KAZAN-AGG`
- Geometry: Polygon (the buffer) in WGS84 (EPSG:4326)
"""

import json
from pathlib import Path

import pyproj
from shapely.geometry import Point, mapping
from shapely.ops import transform

KAZAN_LAT = 55.7887
KAZAN_LON = 49.1221
RADIUS_KM = 30.0
SHAPE_ISO = "RU-KAZAN-AGG"
SHAPE_NAME = "Kazan Agglomeration (center + 30 km buffer)"
OUT_PATH = Path("data/raw/regions/kazan-agglomeration.geojson")


def main() -> None:
    to_utm = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32639", always_xy=True).transform
    to_wgs = pyproj.Transformer.from_crs("EPSG:32639", "EPSG:4326", always_xy=True).transform

    center_wgs = Point(KAZAN_LON, KAZAN_LAT)
    center_utm = transform(to_utm, center_wgs)
    buffer_utm = center_utm.buffer(RADIUS_KM * 1000.0)
    buffer_wgs = transform(to_wgs, buffer_utm)

    feature_collection = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "shapeISO": SHAPE_ISO,
                    "shapeName": SHAPE_NAME,
                    "radius_km": RADIUS_KM,
                    "center_lat": KAZAN_LAT,
                    "center_lon": KAZAN_LON,
                },
                "geometry": mapping(buffer_wgs),
            }
        ],
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        json.dump(feature_collection, f)

    print(f"Wrote {OUT_PATH}")
    print(f"Bounds (lon_min, lat_min, lon_max, lat_max): {buffer_wgs.bounds}")


if __name__ == "__main__":
    main()
