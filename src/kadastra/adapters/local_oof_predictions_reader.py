"""Local-filesystem adapter for the OOF predictions reader port.

Picks the right run directory and OOF artifact for the requested
model (ADR-0016 quartet support):

- ``catboost``: scans both ``catboost-object-{class}_<ts>`` (legacy,
  artifact ``oof_predictions.parquet``) and
  ``quartet-object-{class}_<ts>`` (artifact
  ``catboost_oof_predictions.parquet``) and picks the run with the
  most recent timestamp suffix.
- ``ebm`` / ``grey_tree`` / ``naive_linear``: only quartet runs;
  reads ``{model}_oof_predictions.parquet``.

Returns an empty typed DataFrame on missing path / missing run /
missing artifact — never raises.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from kadastra.domain.asset_class import AssetClass

_LEGACY_ARTIFACT = "oof_predictions.parquet"
_LEGACY_PREFIX_TPL = "catboost-object-{class_}_"
_QUARTET_PREFIX_TPL = "quartet-object-{class_}_"
_QUARTET_MODELS = {"catboost", "ebm", "grey_tree", "naive_linear"}
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

    def load_latest(self, asset_class: AssetClass, *, model: str = "catboost") -> pl.DataFrame:
        if model not in _QUARTET_MODELS:
            raise ValueError(f"unknown model {model!r}; expected one of {_QUARTET_MODELS}")
        if not self._base_path.is_dir():
            return pl.DataFrame(schema=_OOF_SCHEMA)

        run_artifact_pairs = self._enumerate_runs(asset_class.value, model)
        if not run_artifact_pairs:
            return pl.DataFrame(schema=_OOF_SCHEMA)
        # Sort by run dir name (timestamp suffix is lexicographically
        # ordered), pick the most recent run with an existing artifact.
        run_artifact_pairs.sort(key=lambda p: p[0].name)
        for run_dir, artifact_name in reversed(run_artifact_pairs):
            path = run_dir / artifact_name
            if path.is_file():
                return pl.read_parquet(path)
        return pl.DataFrame(schema=_OOF_SCHEMA)

    def _enumerate_runs(self, class_value: str, model: str) -> list[tuple[Path, str]]:
        """Return (run_dir, artifact_filename) pairs eligible for model."""
        legacy_prefix = _LEGACY_PREFIX_TPL.format(class_=class_value)
        quartet_prefix = _QUARTET_PREFIX_TPL.format(class_=class_value)
        pairs: list[tuple[Path, str]] = []
        for entry in self._base_path.iterdir():
            if not entry.is_dir():
                continue
            if model == "catboost" and entry.name.startswith(legacy_prefix):
                pairs.append((entry, _LEGACY_ARTIFACT))
            elif entry.name.startswith(quartet_prefix):
                pairs.append((entry, f"{model}_oof_predictions.parquet"))
        return pairs
