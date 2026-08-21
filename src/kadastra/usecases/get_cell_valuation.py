"""Read the ADR-0029 cell valuation layer for the map UI's
«Стоимость типового объекта» mode.

Mirror of :class:`GetCellTsorf`, but reads from
:class:`ParquetCellValuationStore` (gold cell valuation keyed by
``(h3_index, reference_variant)``). The map UI colours res cells by the
predicted price of a reference object — ``reference_rub_per_m2`` — or by
the pure location score, dimming cells without sample coverage
(``sample_covered=False``).
"""

from __future__ import annotations

from kadastra.adapters.parquet_cell_valuation_store import ParquetCellValuationStore
from kadastra.domain.asset_class import AssetClass

# The two colourable metrics of the cell valuation layer. Drives the
# frontend's metric selector and the API's ``metric`` query param.
CELL_VALUATION_METRICS: tuple[str, ...] = ("reference", "location_score")


class GetCellValuation:
    def __init__(self, store: ParquetCellValuationStore) -> None:
        self._store = store

    def execute(
        self,
        region_code: str,
        asset_class: AssetClass,
        variant: str,
        metric: str,
    ) -> list[dict[str, object]]:
        """``[{"hex", "value", "covered"}, …]`` for one (variant, metric).

        Stub — replaced by the implementation commit (TDD).
        """
        raise NotImplementedError

    def variant_map(self, region_code: str) -> dict[str, list[str]]:
        """``{asset_class: [reference_variant, …]}`` for every asset class.

        Stub — replaced by the implementation commit (TDD).
        """
        raise NotImplementedError

    def get_cell_detail(
        self,
        region_code: str,
        asset_class: AssetClass,
        h3_index: str,
    ) -> dict[str, dict[str, object]]:
        """``{variant: {reference_rub_per_m2, …}}`` for one cell.

        Stub — replaced by the implementation commit (TDD).
        """
        raise NotImplementedError
