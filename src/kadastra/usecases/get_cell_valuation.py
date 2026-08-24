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

import json

import polars as pl

from kadastra.adapters.parquet_cell_valuation_store import ParquetCellValuationStore
from kadastra.domain.asset_class import AssetClass

# The two colourable metrics of the cell valuation layer. Drives the
# frontend's metric selector and the API's ``metric`` query param.
CELL_VALUATION_METRICS: tuple[str, ...] = ("reference", "location_score")

# API metric slug → gold column. Kept as a mapping (not f-string
# interpolation) so a typoed slug is a clean KeyError, never a column
# lookup against an attacker-controlled name.
_METRIC_COLUMNS: dict[str, str] = {
    "reference": "reference_rub_per_m2",
    "location_score": "location_score_rub_per_m2",
}

# Per-cell detail payload: everything except the identity columns
# (h3_index / lat / lon / reference_variant — the latter is the dict key).
# ``top_terms_json`` is parsed into a ``top_terms`` list before the
# payload leaves the usecase; older parquets without the column degrade
# to no-terms gracefully.
_DETAIL_COLUMNS: tuple[str, ...] = (
    "reference_rub_per_m2",
    "location_score_rub_per_m2",
    "n_sample_objects",
    "sample_covered",
    "top_terms_json",
)


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

        ``metric`` is ``reference`` (predicted reference-object price) or
        ``location_score`` (pure location component). Unknown metric or
        variant → KeyError (the API surfaces 400); missing partition →
        FileNotFoundError (the API surfaces 404).
        """
        column = self._metric_column(metric)
        df = self._store.load(region_code, asset_class)
        available = sorted(df["reference_variant"].unique().to_list())
        if variant not in available:
            raise KeyError(
                f"reference_variant {variant!r} not in asset_class={asset_class.value!r}; available: {available}"
            )
        slim = df.filter(pl.col("reference_variant") == variant).select(
            [
                "h3_index",
                pl.col(column).alias("value"),
                pl.col("sample_covered").alias("covered"),
            ]
        )
        return [
            {"hex": r["h3_index"], "value": r["value"], "covered": r["covered"]} for r in slim.iter_rows(named=True)
        ]

    def variant_map(self, region_code: str) -> dict[str, list[str]]:
        """``{asset_class: [reference_variant, …]}`` for every asset class.

        One call populates the frontend's ВРИ selector; classes without a
        built partition map to ``[]`` so the UI shows an empty selector,
        not an error."""
        return {ac.value: self._list_variants(region_code, ac) for ac in AssetClass}

    def get_cell_detail(
        self,
        region_code: str,
        asset_class: AssetClass,
        h3_index: str,
    ) -> dict[str, dict[str, object]]:
        """``{variant: {reference_rub_per_m2, location_score_rub_per_m2,
        n_sample_objects, sample_covered, top_terms?}}`` for one cell —
        every variant (landplot's ВРИ alternatives included) side by side.
        ``top_terms`` (top locational EBM contributions) is stored on the
        «default» variant only — the decomposition is variant-independent.

        Feeds the cell inspector panel when clicking on a hex in
        «Стоимость типового объекта» mode. Missing partition →
        FileNotFoundError; a cell absent from the partition → ``{}``.
        """
        df = self._store.load(region_code, asset_class)
        rows = df.filter(pl.col("h3_index") == h3_index)
        columns = [c for c in _DETAIL_COLUMNS if c in df.columns]
        out: dict[str, dict[str, object]] = {}
        for r in rows.iter_rows(named=True):
            entry = {c: r[c] for c in columns}
            raw_terms = entry.pop("top_terms_json", None)
            if raw_terms:
                entry["top_terms"] = json.loads(str(raw_terms))
            out[str(r["reference_variant"])] = entry
        return out

    def _list_variants(self, region_code: str, asset_class: AssetClass) -> list[str]:
        try:
            df = self._store.load(region_code, asset_class)
        except FileNotFoundError:
            return []
        return sorted(df["reference_variant"].unique().to_list())

    @staticmethod
    def _metric_column(metric: str) -> str:
        try:
            return _METRIC_COLUMNS[metric]
        except KeyError:
            raise KeyError(f"unknown metric: {metric!r}; expected one of {CELL_VALUATION_METRICS}") from None
