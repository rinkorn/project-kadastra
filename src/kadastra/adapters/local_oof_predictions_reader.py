"""Local-filesystem adapter for the OOF predictions reader port.

Looks for run directories named ``catboost-object-{class}_<timestamp>``
under the configured registry base path, picks the lexicographically
largest one (timestamp suffix is sortable), and reads
``oof_predictions.parquet`` from inside.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from kadastra.domain.asset_class import AssetClass

_ARTIFACT_NAME = "oof_predictions.parquet"
_OOF_SCHEMA = {
    "object_id": pl.Utf8,
    "lat": pl.Float64,
    "lon": pl.Float64,
    "fold_id": pl.Int64,
    "y_true": pl.Float64,
    "y_pred_oof": pl.Float64,
}


class LocalOofPredictionsReader:
    def __init__(self, base_path: Path) -> None:
        self._base_path = base_path

    def load_latest(self, asset_class: AssetClass) -> pl.DataFrame:
        prefix = f"catboost-object-{asset_class.value}_"
        if not self._base_path.is_dir():
            return pl.DataFrame(schema=_OOF_SCHEMA)
        matches = sorted(
            d.name
            for d in self._base_path.iterdir()
            if d.is_dir() and d.name.startswith(prefix)
        )
        if not matches:
            return pl.DataFrame(schema=_OOF_SCHEMA)
        latest = matches[-1]
        path = self._base_path / latest / _ARTIFACT_NAME
        if not path.is_file():
            return pl.DataFrame(schema=_OOF_SCHEMA)
        return pl.read_parquet(path)
