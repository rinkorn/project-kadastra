"""Port for loading the trained EBM (White Box) model per class.

The EBM is the only quartet member that can explain its own predictions
(interpret-ml shape functions), so it gets its own loader port separate
from the CatBoost-specific ``ModelLoaderPort``.
"""

from __future__ import annotations

from typing import Any, Protocol

from kadastra.domain.asset_class import AssetClass


class EbmExplainerPort(Protocol):
    """The subset of ``EbmQuartetModel`` used for local explanations."""

    def explain(self, X: Any, feature_names: list[str]) -> dict[str, Any]: ...


class EbmModelLoaderPort(Protocol):
    def load_latest(self, asset_class: AssetClass) -> EbmExplainerPort: ...
