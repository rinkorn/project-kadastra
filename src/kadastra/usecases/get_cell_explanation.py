"""GetCellExplanation use case — on-demand FULL locational EBM
decomposition for one cell (ADR-0029).

The precomputed cell_valuation store keeps only the top-15 locational
terms per cell (storage economy — see ``BuildCellValuation._TOP_TERMS_N``).
This use case recomputes the complete term list for a single cell on
request, reusing the exact build-time assembly path
(:func:`load_cell_feature_frame` / :func:`assemble_cell_frame`) and the
training-time parent aggregates, so the feature vector matches what
``BuildCellValuation`` scored. Powers the cell inspector's «Показать
все термы» toggle.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from kadastra.domain.asset_class import AssetClass
from kadastra.etl.cell_reference_object import build_reference_objects
from kadastra.etl.relative_features import compute_parent_aggregates
from kadastra.ml.cell_location_terms import sum_location_terms, top_location_terms
from kadastra.ml.object_feature_columns import select_object_feature_columns
from kadastra.ml.object_feature_matrix import build_object_feature_matrix
from kadastra.ports.ebm_model_loader import EbmScorerLoaderPort
from kadastra.ports.feature_reader import FeatureReaderPort
from kadastra.ports.valuation_object_reader import ValuationObjectReaderPort
from kadastra.usecases.build_cell_valuation import (
    assemble_cell_frame,
    load_cell_feature_frame,
)

_TARGET_COLUMN = "synthetic_target_rub_per_m2"


class GetCellExplanation:
    def __init__(
        self,
        cell_feature_reader: FeatureReaderPort,
        object_reader: ValuationObjectReaderPort,
        ebm_loader: EbmScorerLoaderPort,
        *,
        asset_classes: list[AssetClass],
        cell_feature_sets: tuple[str, ...],
        resolution: int,
        relative_parent_resolutions: list[int],
        relative_feature_columns: list[str],
        current_year: int,
    ) -> None:
        self._cell_feature_reader = cell_feature_reader
        self._object_reader = object_reader
        self._ebm_loader = ebm_loader
        self._asset_classes = asset_classes
        self._cell_feature_sets = cell_feature_sets
        self._resolution = resolution
        self._relative_parent_resolutions = relative_parent_resolutions
        self._relative_feature_columns = relative_feature_columns
        self._current_year = current_year

    def explain(
        self,
        region_code: str,
        asset_class: AssetClass,
        h3_index: str,
    ) -> dict[str, Any] | None:
        """Full locational EBM decomposition for one cell.

        Returns ``{intercept, location_score, terms}`` where ``terms``
        lists EVERY locational term (``{feature, contribution}``, sorted
        by descending |contribution|), so
        ``intercept + Σ contribution == location_score`` exactly.
        ``None`` when the cell, the class gold, or the EBM artifact is
        missing.
        """
        cells = load_cell_feature_frame(
            self._cell_feature_reader,
            region_code,
            self._resolution,
            self._cell_feature_sets,
        )
        row = cells.filter(pl.col("h3_index") == h3_index)
        if row.is_empty():
            return None

        # Same training-scope gold + cross-class parent aggregates the
        # batch build used (BuildCellValuation.execute).
        gold = {
            ac: self._object_reader.load(region_code, ac).drop_nulls(subset=[_TARGET_COLUMN])
            for ac in self._asset_classes
        }
        class_gold = gold[asset_class]
        if class_gold.is_empty():
            return None
        non_empty = [df for df in gold.values() if not df.is_empty()]
        combined = pl.concat(non_empty, how="vertical_relaxed")
        rel_columns = [c for c in self._relative_feature_columns if c in combined.columns]
        aggregates = compute_parent_aggregates(
            combined,
            parent_resolutions=self._relative_parent_resolutions,
            feature_columns=rel_columns,
        )

        numeric_cols, categorical_cols = select_object_feature_columns(class_gold)
        feature_names = numeric_cols + categorical_cols
        # The decomposition is variant-independent (locational terms
        # don't touch template attributes) — the default template is
        # enough.
        template = build_reference_objects(class_gold, current_year=self._current_year)[0]
        frame = assemble_cell_frame(
            row,
            template,
            class_gold.schema,
            aggregates,
            parent_resolutions=self._relative_parent_resolutions,
            rel_columns=rel_columns,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
        )
        X = build_object_feature_matrix(
            frame,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
        )
        try:
            model = self._ebm_loader.load_latest(asset_class)
        except FileNotFoundError:
            return None

        terms = np.asarray(model.eval_terms(X), dtype=np.float64)
        term_features = [tuple(feature_names[i] for i in idxs) for idxs in model.term_feature_indices()]
        intercept = float(model.intercept())
        location_score = float(sum_location_terms(terms, term_features, model.intercept())[0])
        # top_location_terms with top_n ≥ #locational terms returns the
        # full locational list, sorted and rounded — no separate code path.
        all_terms = top_location_terms(terms, term_features, top_n=terms.shape[1])[0]
        return {
            "intercept": intercept,
            "location_score": location_score,
            "terms": all_terms,
        }
