"""Pure functions to flatten one NSPD GeoJSON feature into a row dict.

Each input is a single ``Feature`` as it appears on disk in
``data/raw/nspd/{buildings,landplots}-kazan/page-NNNN.json`` (the value
of ``data["data"]["features"][i]``). The output is a flat ``dict``
ready to be appended to a polars frame:

- atomic options (cad_num, area, cost_value, cost_index, ...) are
  unwrapped from ``properties.options``,
- centroid is computed in EPSG:3857 and projected to WGS84,
- the original polygon is preserved as WKT (in EPSG:3857) for any
  subsequent spatial filtering that doesn't want to redo the projection.
"""

from __future__ import annotations

from typing import Any

from pyproj import Transformer
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

from kadastra.domain.asset_class import AssetClass
from kadastra.domain.classify_nspd_purpose import classify_nspd_building_purpose

_TRANSFORMER_3857_TO_4326 = Transformer.from_crs(
    "EPSG:3857", "EPSG:4326", always_xy=True
)


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return int(s)
        except ValueError:
            try:
                return int(float(s))
            except ValueError:
                return None
    if isinstance(value, float):
        return int(value)
    return None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _to_str(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


def _centroid_wgs84(geometry_dict: dict[str, Any]) -> tuple[float, float]:
    geom: BaseGeometry = shape(geometry_dict)
    centroid = geom.centroid
    lon, lat = _TRANSFORMER_3857_TO_4326.transform(centroid.x, centroid.y)
    return lat, lon


def _polygon_wkt(geometry_dict: dict[str, Any]) -> str:
    return shape(geometry_dict).wkt


def parse_nspd_building_feature(feature: dict[str, Any]) -> dict[str, Any]:
    options = feature["properties"]["options"]
    purpose_raw = _to_str(options.get("purpose"))
    asset_class = classify_nspd_building_purpose(purpose_raw)
    lat, lon = _centroid_wgs84(feature["geometry"])

    return {
        "geom_data_id": int(feature["id"]),
        "cad_num": _to_str(options.get("cad_num")),
        "purpose": purpose_raw,
        "asset_class": asset_class.value if asset_class is not None else None,
        "lat": lat,
        "lon": lon,
        "area_m2": _to_float(options.get("build_record_area")),
        "cost_value_rub": _to_float(options.get("cost_value")),
        "cost_index_rub_per_m2": _to_float(options.get("cost_index")),
        "year_built": _to_int(options.get("year_built")),
        "floors": _to_int(options.get("floors")),
        "underground_floors": _to_int(options.get("underground_floors")),
        "materials": _to_str(options.get("materials")),
        "ownership_type": _to_str(options.get("ownership_type")),
        "registration_date": _to_str(options.get("build_record_registration_date")),
        "readable_address": _to_str(options.get("readable_address")),
        "polygon_wkt_3857": _polygon_wkt(feature["geometry"]),
    }


def parse_nspd_landplot_feature(feature: dict[str, Any]) -> dict[str, Any]:
    options = feature["properties"]["options"]
    lat, lon = _centroid_wgs84(feature["geometry"])

    return {
        "geom_data_id": int(feature["id"]),
        "cad_num": _to_str(options.get("cad_num")),
        "asset_class": AssetClass.LANDPLOT.value,
        "lat": lat,
        "lon": lon,
        "area_m2": _to_float(options.get("specified_area")),
        "cost_value_rub": _to_float(options.get("cost_value")),
        "cost_index_rub_per_m2": _to_float(options.get("cost_index")),
        "land_record_category_type": _to_str(options.get("land_record_category_type")),
        "land_record_subtype": _to_str(options.get("land_record_subtype")),
        "ownership_type": _to_str(options.get("ownership_type")),
        "registration_date": _to_str(options.get("land_record_reg_date")),
        "readable_address": _to_str(options.get("readable_address")),
        "polygon_wkt_3857": _polygon_wkt(feature["geometry"]),
    }
