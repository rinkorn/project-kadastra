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


class EbmScorerPort(Protocol):
    """The subset of ``EbmQuartetModel`` used by the cell valuation layer
    (ADR-0029): batch prediction + additive term decomposition."""

    def predict(self, X: Any) -> Any: ...

    def eval_terms(self, X: Any) -> Any: ...

    def intercept(self) -> float: ...

    def term_feature_indices(self) -> list[tuple[int, ...]]: ...


class EbmModelLoaderPort(Protocol):
    def load_latest(self, asset_class: AssetClass) -> EbmExplainerPort: ...


class EbmScorerLoaderPort(Protocol):
    """Loader port returning the richer scoring interface (ADR-0029).

    Structural protocol — ``LocalEbmModelLoader`` satisfies both this
    and :class:`EbmModelLoaderPort` without code changes.
    """

    def load_latest(self, asset_class: AssetClass) -> EbmScorerPort: ...
