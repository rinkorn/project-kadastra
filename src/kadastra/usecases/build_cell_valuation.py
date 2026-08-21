"""BuildCellValuation use case (ADR-0029).

Scores every anchor cell (res 10, Слой 1) with the class's EBM quartet
model in two products:

1. **reference price** — the EBM prediction for the class's reference
   object (median/mode template from gold) placed in the cell;
   landplot builds one template per top-N ``vri``.
2. **location score** — ``intercept + Σ locational EBM terms``
   (object and mixed terms dropped; NOT a price prediction).

Relative features use the fixed training-time gold aggregates
(:func:`compute_parent_aggregates` / :func:`join_relative_features`),
so a cell's ``*_rel_p*`` values are measured against the same parent
distribution the model was trained on.

Output per class: ``data/gold/cell_valuation/region=…/asset_class=…``
via the injected store port, one row per (cell, reference_variant),
with per-cell sample-coverage flags (ADR-0028 linkage).
"""

from __future__ import annotations

from dataclasses import dataclass

from kadastra.domain.asset_class import AssetClass
from kadastra.etl.cell_reference_object import ReferenceObject
from kadastra.ports.ebm_model_loader import EbmScorerLoaderPort
from kadastra.ports.feature_reader import FeatureReaderPort
from kadastra.ports.valuation_object_reader import ValuationObjectReaderPort
from kadastra.ports.valuation_object_store import ValuationObjectStorePort


@dataclass(frozen=True)
class CellValuationClassResult:
    """Per-class summary of a build run (for _meta.json / sanity report)."""

    asset_class: AssetClass
    n_cells: int
    reference_objects: list[ReferenceObject]
    cbd_correlation: float | None
    covered_share: float


class BuildCellValuation:
    def __init__(
        self,
        cell_feature_reader: FeatureReaderPort,
        object_reader: ValuationObjectReaderPort,
        ebm_loader: EbmScorerLoaderPort,
        output_store: ValuationObjectStorePort,
        *,
        cell_feature_sets: tuple[str, ...],
        resolution: int,
        relative_parent_resolutions: list[int],
        relative_feature_columns: list[str],
        current_year: int,
        landplot_vri_top_n: int = 5,
    ) -> None:
        self._cell_feature_reader = cell_feature_reader
        self._object_reader = object_reader
        self._ebm_loader = ebm_loader
        self._output_store = output_store
        self._cell_feature_sets = cell_feature_sets
        self._resolution = resolution
        self._relative_parent_resolutions = relative_parent_resolutions
        self._relative_feature_columns = relative_feature_columns
        self._current_year = current_year
        self._landplot_vri_top_n = landplot_vri_top_n

    def execute(
        self,
        region_code: str,
        asset_classes: list[AssetClass],
    ) -> dict[AssetClass, CellValuationClassResult]:
        raise NotImplementedError
