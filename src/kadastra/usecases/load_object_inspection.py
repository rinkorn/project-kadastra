"""Per-object inspection loader for the map UI.

Joins gold valuation objects with OOF predictions (via
``OofPredictionsReaderPort``) so the API/UI can compare ЕГРН target
against the spatial-CV-honest prediction for every object — without
re-running training.

Two read shapes:

- ``list_for_map(region, asset_class)`` — lightweight payload for the
  scatter layer: ``{object_id, lat, lon, y_true, y_pred_oof,
  residual, fold_id, polygon_wkt_3857}`` per object. Polygon WKT is
  shipped raw so the API edge can convert to GeoJSON-WGS84 once per
  request without coupling the usecase to deck.gl/maplibre concerns.
- ``get_detail(region, asset_class, object_id)`` — full feature dict
  for one object (every gold column + OOF columns).

If no OOF artifact exists for the class, prediction columns are nulls
and ``residual`` is null too — caller can still render y_true.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from kadastra.domain.asset_class import AssetClass
from kadastra.ports.oof_predictions_reader import OofPredictionsReaderPort
from kadastra.ports.valuation_object_reader import ValuationObjectReaderPort

_TARGET = "synthetic_target_rub_per_m2"
# ADR-0016 quartet — order matters only for stable iteration in
# ``get_detail_quartet`` (UI shows them in this order in the
# comparison panel).
_QUARTET_MODELS = ("catboost", "ebm", "grey_tree", "naive_linear")
# Curated per-object numeric features the map UI can colour the
# object scatter by. Same naming as the gold parquet — the slim
# payload only includes columns actually present on the input frame
# (per-class schema variation, e.g. landplot has no ``flats``).
OBJECT_FEATURE_COLUMNS: tuple[str, ...] = (
    # Building / land descriptors
    "area_m2", "levels", "flats", "year_built", "age_years",
    # ADR-0019 distance to nearest geometry — polygonal / linear / POI
    "dist_to_water_m", "dist_to_park_m", "dist_to_industrial_m",
    "dist_to_cemetery_m", "dist_to_landfill_m",
    "dist_to_powerline_m", "dist_to_railway_m",
    "dist_to_school_m", "dist_to_kindergarten_m", "dist_to_clinic_m",
    "dist_to_hospital_m", "dist_to_pharmacy_m", "dist_to_supermarket_m",
    "dist_to_cafe_m", "dist_to_restaurant_m",
    "dist_to_bus_stop_m", "dist_to_tram_stop_m", "dist_to_railway_station_m",
    "dist_metro_m", "dist_entrance_m",
    # Polygonal share-in-buffer at 500 m
    "water_share_500m", "park_share_500m",
    "industrial_share_500m", "cemetery_share_500m",
    # Road density + zonal counts at 500 m / 1 km
    "road_length_500m",
    "count_stations_1km", "count_entrances_500m",
    "count_apartments_500m", "count_houses_500m", "count_commercial_500m",
    "school_within_500m", "kindergarten_within_500m",
    "clinic_within_500m", "hospital_within_500m",
    "pharmacy_within_500m", "supermarket_within_500m",
    "cafe_within_500m", "restaurant_within_500m",
    "bus_stop_within_500m", "tram_stop_within_500m",
    "railway_station_within_500m",
)


class LoadObjectInspection:
    def __init__(
        self,
        reader: ValuationObjectReaderPort,
        oof_reader: OofPredictionsReaderPort,
    ) -> None:
        self._reader = reader
        self._oof_reader = oof_reader

    def list_for_map(
        self,
        region_code: str,
        asset_class: AssetClass,
        *,
        model: str = "catboost",
    ) -> list[dict[str, Any]]:
        joined = self._load_joined(region_code, asset_class, model=model)
        if joined.is_empty():
            return []
        core_cols = [
            "object_id", "lat", "lon",
            "y_true", "y_pred_oof", "residual", "fold_id",
            "polygon_wkt_3857",
        ]
        # Curated features are optional: a per-class schema may be
        # missing some columns (landplot has no ``flats``, an older
        # parquet may not yet carry POI distances), and the slim
        # payload simply omits them when absent.
        feature_cols = [c for c in OBJECT_FEATURE_COLUMNS if c in joined.columns]
        slim = joined.select(core_cols + feature_cols)
        return slim.to_dicts()

    def get_detail(
        self,
        region_code: str,
        asset_class: AssetClass,
        object_id: str,
        *,
        model: str = "catboost",
    ) -> dict[str, Any] | None:
        joined = self._load_joined(region_code, asset_class, model=model)
        if joined.is_empty():
            return None
        match = joined.filter(pl.col("object_id") == object_id)
        if match.is_empty():
            return None
        return match.row(0, named=True)

    def get_detail_quartet(
        self,
        region_code: str,
        asset_class: AssetClass,
        object_id: str,
    ) -> dict[str, Any] | None:
        objects = self._reader.load(region_code, asset_class)
        if objects.is_empty():
            return None
        match = objects.filter(pl.col("object_id") == object_id)
        if match.is_empty():
            return None
        gold_row = match.rename({_TARGET: "y_true"}).row(0, named=True)
        y_true = gold_row.get("y_true")

        models_payload: dict[str, dict[str, Any]] = {}
        for model in _QUARTET_MODELS:
            entry: dict[str, Any] = {
                "y_pred_oof": None,
                "residual": None,
                "fold_id": None,
            }
            oof = self._oof_reader.load_latest(asset_class, model=model)
            if not oof.is_empty() and "object_id" in oof.columns:
                sub = oof.filter(pl.col("object_id") == object_id)
                if not sub.is_empty():
                    row = sub.row(0, named=True)
                    y_pred = row.get("y_pred_oof")
                    entry["y_pred_oof"] = y_pred
                    entry["fold_id"] = row.get("fold_id")
                    if y_pred is not None and y_true is not None:
                        entry["residual"] = y_pred - y_true
            models_payload[model] = entry

        gold_row["models"] = models_payload
        return gold_row

    def _load_joined(
        self, region_code: str, asset_class: AssetClass, *, model: str
    ) -> pl.DataFrame:
        objects = self._reader.load(region_code, asset_class)
        if objects.is_empty():
            return objects

        oof = self._oof_reader.load_latest(asset_class, model=model)
        renamed_target = objects.rename({_TARGET: "y_true"})

        if oof.is_empty() or "object_id" not in oof.columns:
            return renamed_target.with_columns(
                pl.lit(None, dtype=pl.Float64).alias("y_pred_oof"),
                pl.lit(None, dtype=pl.Int64).alias("fold_id"),
                pl.lit(None, dtype=pl.Float64).alias("residual"),
            )

        right = oof.select(["object_id", "fold_id", "y_pred_oof"])
        return renamed_target.join(right, on="object_id", how="left").with_columns(
            (pl.col("y_pred_oof") - pl.col("y_true")).alias("residual")
        )
