"""Load the latest trained EBM model for a class from the local registry.

Scans ``quartet-object-{class}_<ts>/ebm_model.pkl`` and deserializes the
most recent one. Raises ``FileNotFoundError`` when no such artifact exists.
"""

from __future__ import annotations

from pathlib import Path

from kadastra.adapters.ebm_quartet_model import EbmQuartetModel
from kadastra.domain.asset_class import AssetClass


class LocalEbmModelLoader:
    def __init__(self, base_path: Path) -> None:
        self._base_path = base_path

    def load_latest(self, asset_class: AssetClass) -> EbmQuartetModel:
        prefix = f"quartet-object-{asset_class.value}_"
        if not self._base_path.is_dir():
            raise FileNotFoundError(f"registry path does not exist: {self._base_path}")
        runs = sorted(d for d in self._base_path.iterdir() if d.is_dir() and d.name.startswith(prefix))
        for run in reversed(runs):
            path = run / "ebm_model.pkl"
            if path.is_file():
                return EbmQuartetModel.deserialize(path.read_bytes())
        raise FileNotFoundError(f"no ebm_model.pkl for asset_class={asset_class.value}")
