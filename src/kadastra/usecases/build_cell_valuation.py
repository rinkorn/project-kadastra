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
from typing import cast

import numpy as np
import polars as pl

from kadastra.domain.asset_class import AssetClass
from kadastra.etl.cell_reference_object import ReferenceObject, build_reference_objects
from kadastra.etl.h3_coverage import add_h3_index, h3_cells_to_latlng
from kadastra.etl.relative_features import compute_parent_aggregates, join_relative_features
from kadastra.ml.cell_location_terms import sum_location_terms
from kadastra.ml.object_feature_columns import select_object_feature_columns
from kadastra.ml.object_feature_matrix import build_object_feature_matrix
from kadastra.ports.ebm_model_loader import EbmScorerLoaderPort
from kadastra.ports.feature_reader import FeatureReaderPort
from kadastra.ports.valuation_object_reader import ValuationObjectReaderPort
from kadastra.ports.valuation_object_store import ValuationObjectStorePort

_TARGET_COLUMN = "synthetic_target_rub_per_m2"

# Слой 1 sets joined for scoring: the six ADR-0027 sets plus the
# ADR-0029 ``enrichment`` set (ADR-0021..0025 features per cell).
CELL_VALUATION_FEATURE_SETS: tuple[str, ...] = (
    "geom_distance",
    "poly_area",
    "zonal",
    "road_density",
    "metro",
    "walk_dist",
    "enrichment",
)


def _infer_literal_dtype(value: object) -> pl.DataType | type[pl.DataType]:
    """Literal dtype for a template attribute absent from the gold schema."""
    if isinstance(value, str):
        return pl.Utf8
    if isinstance(value, bool):
        return pl.Boolean
    if isinstance(value, int):
        return pl.Int64
    return pl.Float64


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
        cells = self._load_cell_frame(region_code)

        # Training-scope gold: same drop-nulls filter TrainQuartet used.
        gold = {
            ac: self._object_reader.load(region_code, ac).drop_nulls(subset=[_TARGET_COLUMN]) for ac in asset_classes
        }
        non_empty = [df for df in gold.values() if not df.is_empty()]
        combined = pl.concat(non_empty, how="vertical_relaxed") if non_empty else pl.DataFrame()
        rel_columns = [c for c in self._relative_feature_columns if c in combined.columns]
        aggregates = compute_parent_aggregates(
            combined,
            parent_resolutions=self._relative_parent_resolutions,
            feature_columns=rel_columns,
        )

        results: dict[AssetClass, CellValuationClassResult] = {}
        for ac in asset_classes:
            class_gold = gold[ac]
            numeric_cols, categorical_cols = select_object_feature_columns(class_gold)
            feature_names = numeric_cols + categorical_cols
            model = self._ebm_loader.load_latest(ac)
            templates = build_reference_objects(
                class_gold,
                current_year=self._current_year,
                vri_top_n=self._landplot_vri_top_n if ac == AssetClass.LANDPLOT else None,
            )

            location_score: np.ndarray | None = None
            variant_frames: list[pl.DataFrame] = []
            for template in templates:
                frame = self._assemble_cell_frame(
                    cells,
                    template,
                    class_gold,
                    aggregates,
                    rel_columns,
                    numeric_cols,
                    categorical_cols,
                )
                X = build_object_feature_matrix(
                    frame,
                    numeric_cols=numeric_cols,
                    categorical_cols=categorical_cols,
                )
                preds = np.asarray(model.predict(X), dtype=np.float64)
                if location_score is None:
                    terms = np.asarray(model.eval_terms(X), dtype=np.float64)
                    term_features = [tuple(feature_names[i] for i in idxs) for idxs in model.term_feature_indices()]
                    location_score = sum_location_terms(terms, term_features, model.intercept())
                variant_frames.append(
                    cells.select("h3_index", "lat", "lon").with_columns(
                        pl.lit(template.variant, dtype=pl.Utf8).alias("reference_variant"),
                        pl.Series("reference_rub_per_m2", preds, dtype=pl.Float64),
                        pl.Series("location_score_rub_per_m2", location_score, dtype=pl.Float64),
                    )
                )

            out = pl.concat(variant_frames)
            out = self._join_sample_coverage(out, class_gold)
            self._output_store.save(region_code, ac, out)

            assert location_score is not None  # at least the default template exists
            covered_mean = out["sample_covered"].mean()
            results[ac] = CellValuationClassResult(
                asset_class=ac,
                n_cells=cells.height,
                reference_objects=templates,
                cbd_correlation=self._cbd_correlation(cells, location_score),
                covered_share=float(cast("float", covered_mean)) if covered_mean is not None else 0.0,
            )
        return results

    def _load_cell_frame(self, region_code: str) -> pl.DataFrame:
        """Join all Слой 1 feature sets on ``h3_index`` and add lat/lon."""
        cells: pl.DataFrame | None = None
        for feature_set in self._cell_feature_sets:
            part = self._cell_feature_reader.load(region_code, self._resolution, feature_set)
            if "resolution" in part.columns:
                part = part.drop("resolution")
            cells = part if cells is None else cells.join(part, on="h3_index", how="full", coalesce=True)
        if cells is None:
            raise ValueError("BuildCellValuation: no cell feature sets configured")
        coords = h3_cells_to_latlng(cells["h3_index"].to_list())
        return cells.with_columns(
            pl.Series("lat", [c[0] for c in coords], dtype=pl.Float64),
            pl.Series("lon", [c[1] for c in coords], dtype=pl.Float64),
        )

    def _assemble_cell_frame(
        self,
        cells: pl.DataFrame,
        template: ReferenceObject,
        class_gold: pl.DataFrame,
        aggregates: pl.DataFrame,
        rel_columns: list[str],
        numeric_cols: list[str],
        categorical_cols: list[str],
    ) -> pl.DataFrame:
        """Cell location features + template object attributes + rel features.

        Every model feature column is guaranteed present: missing ones
        (e.g. a feature set the region lacks) are added as nulls with
        the dtype the gold training frame carried.
        """
        frame = cells
        gold_schema = class_gold.schema
        for col, value in template.attributes.items():
            dtype = gold_schema.get(col) or _infer_literal_dtype(value)
            if col in frame.columns:
                frame = frame.drop(col)
            frame = frame.with_columns(pl.lit(value, dtype=dtype).alias(col))
        present_rel = [c for c in rel_columns if c in frame.columns]
        if present_rel:
            frame = join_relative_features(
                frame,
                aggregates,
                parent_resolutions=self._relative_parent_resolutions,
                feature_columns=present_rel,
            )
        for col in numeric_cols:
            if col not in frame.columns:
                frame = frame.with_columns(pl.lit(None, dtype=pl.Float64).alias(col))
        for col in categorical_cols:
            if col not in frame.columns:
                frame = frame.with_columns(pl.lit(None, dtype=pl.Utf8).alias(col))
        return frame

    def _join_sample_coverage(self, out: pl.DataFrame, class_gold: pl.DataFrame) -> pl.DataFrame:
        """Per-cell training-sample coverage flags (ADR-0028 linkage)."""
        if class_gold.is_empty():
            return out.with_columns(
                pl.lit(0, dtype=pl.Int64).alias("n_sample_objects"),
                pl.lit(False, dtype=pl.Boolean).alias("sample_covered"),
            )
        with_index = add_h3_index(class_gold.select("object_id", "lat", "lon"), resolution=self._resolution)
        counts = with_index.group_by("h3_index").agg(pl.len().cast(pl.Int64).alias("n_sample_objects"))
        joined = out.join(counts, on="h3_index", how="left").with_columns(
            pl.col("n_sample_objects").fill_null(0),
        )
        return joined.with_columns((pl.col("n_sample_objects") > 0).alias("sample_covered"))

    @staticmethod
    def _cbd_correlation(cells: pl.DataFrame, location_score: np.ndarray) -> float | None:
        """Pearson corr(location_score, dist_to_cbd_m) — sanity signal."""
        if "dist_to_cbd_m" not in cells.columns:
            return None
        dist = cells["dist_to_cbd_m"].cast(pl.Float64).to_numpy()
        valid = ~np.isnan(dist) & ~np.isnan(location_score)
        if valid.sum() < 2:
            return None
        corr = np.corrcoef(dist[valid], location_score[valid])[0, 1]
        return float(corr)
