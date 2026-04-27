"""Port for reading OOF (out-of-fold) prediction artifacts.

OOF predictions are written by ``TrainObjectValuationModel`` next to the
trained model.cbm. They contain ``(object_id, lat, lon, fold_id,
y_true, y_pred_oof)`` for every input row, where each prediction came
from a model that did **not** see that row in training (spatial-CV
honest view). The inspector and per-hex aggregator read this artifact
to compare model output against ЕГРН target without retraining.
"""

from __future__ import annotations

from typing import Protocol

import polars as pl

from kadastra.domain.asset_class import AssetClass


class OofPredictionsReaderPort(Protocol):
    def load_latest(self, asset_class: AssetClass, *, model: str = "catboost") -> pl.DataFrame:
        """Return the most recent OOF parquet for the asset class.

        ``model`` selects which adapter-recognized model produced the
        OOF: ``"catboost"`` (default; falls back to legacy
        ``catboost-object-{class}`` runs when no quartet run exists),
        ``"ebm"``, ``"grey_tree"``, ``"naive_linear"`` (all sourced from
        ``quartet-object-{class}`` runs). Schema: ``(object_id: Utf8,
        lat: Float64, lon: Float64, fold_id: Int64, y_true: Float64,
        y_pred_oof: Float64)``.

        Returns an empty (typed) DataFrame if no run / no artifact is
        found — callers should treat that as «predictions unavailable»
        rather than failing.
        """
        ...
