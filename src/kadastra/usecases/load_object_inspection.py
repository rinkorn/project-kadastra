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
        cols = [
            "object_id", "lat", "lon",
            "y_true", "y_pred_oof", "residual", "fold_id",
            "polygon_wkt_3857",
        ]
        slim = joined.select(cols)
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
