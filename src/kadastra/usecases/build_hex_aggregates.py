"""Build per-hex aggregate parquets from per-object gold + OOF predictions.

Output layout (ADR-0016 quartet, per-model):

``{base_path}/region={REGION}/resolution={R}/model={MODEL}/data.parquet``

One row per (h3_index, asset_class) tuple, where ``asset_class`` is
either a real class value or ``"all"`` for the cross-class roll-up.
``MODEL`` is one of ``catboost / ebm / grey_tree / naive_linear`` —
each partition uses the matching ``OofPredictionsReaderPort`` artifact
and produces its own ``median_pred_oof_rub_per_m2`` /
``median_residual_rub_per_m2``. The model-agnostic columns
(``median_target_rub_per_m2``, ``count``, ``dominant_*``) are
duplicated across partitions so the API can read one parquet without
joining.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from kadastra.domain.asset_class import AssetClass
from kadastra.etl.hex_aggregation import aggregate_objects_to_hex
from kadastra.ports.oof_predictions_reader import OofPredictionsReaderPort
from kadastra.ports.valuation_object_reader import ValuationObjectReaderPort


class BuildHexAggregates:
    def __init__(
        self,
        reader: ValuationObjectReaderPort,
        oof_reader: OofPredictionsReaderPort,
        output_base_path: Path,
        resolutions: list[int],
    ) -> None:
        self._reader = reader
        self._oof_reader = oof_reader
        self._output_base_path = output_base_path
        self._resolutions = resolutions

    def execute(
        self,
        region_code: str,
        asset_classes: list[AssetClass],
        *,
        model: str = "catboost",
    ) -> None:
        per_class_frames: list[pl.DataFrame] = []
        for ac in asset_classes:
            objects = self._reader.load(region_code, ac)
            if objects.is_empty():
                continue
            oof = self._oof_reader.load_latest(ac, model=model)
            joined = self._join_oof(objects, oof)
            per_class_frames.append(joined)

        if not per_class_frames:
            return
        combined = pl.concat(per_class_frames, how="vertical_relaxed")

        for resolution in self._resolutions:
            agg = aggregate_objects_to_hex(combined, resolution=resolution)
            out_path = (
                self._output_base_path
                / f"region={region_code}"
                / f"resolution={resolution}"
                / f"model={model}"
                / "data.parquet"
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            agg.write_parquet(out_path)

    @staticmethod
    def _join_oof(objects: pl.DataFrame, oof: pl.DataFrame) -> pl.DataFrame:
        if oof.is_empty() or "object_id" not in oof.columns:
            return objects
        # Bring only the OOF-specific columns; lat/lon/y_true already
        # live on the object frame (the OOF parquet is keyed by
        # object_id and we trust the join, no need to re-import lat/
        # lon/y_true).
        right = oof.select(["object_id", "fold_id", "y_pred_oof"])
        return objects.join(right, on="object_id", how="left")
