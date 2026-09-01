"""Read Слой 1 cell ЦОФ for the map UI's «ЦОФ-сетка» mode.

Mirror of :class:`GetHexAggregates`, but reads from
:class:`ParquetFeatureStore` (Слой 1 feature sets keyed by ``h3_index``)
instead of the object-derived hex-aggregate store. The map UI colours
res-10 cells by a chosen location factor — ``walk_dist_to_school_m``,
``water_share_500m``, ``dist_metro_m``, … — surfacing the grid
infrastructure ADR-0027 built (a demo of «сетка как источник ЦОФ»,
not a valuation function).
"""

from __future__ import annotations

import polars as pl

from kadastra.adapters.parquet_feature_store import ParquetFeatureStore

# The six Слой 1 feature sets built by BuildCell*Features. Listed here
# (not introspected from the store) so a missing partition reads as an
# empty set rather than silently disappearing from the UI.
CELL_TSORF_FEATURE_SETS: tuple[str, ...] = (
    "geom_distance",
    "poly_area",
    "zonal",
    "road_density",
    "metro",
    "walk_dist",
)

# ADR-0029 added the per-cell enrichment set (DEM, road-class,
# isochrones, heritage/ЗОУИТ, ОКТМО-macro, territorial codes) as the
# 7th Слой 1 set. The API surface — the Сет selector, the cell detail
# and /api/feature_options label coverage — spans it too. The
# representativeness report (ADR-0028) deliberately keeps its
# historical six-set scope and still imports CELL_TSORF_FEATURE_SETS.
ENRICHMENT_FEATURE_SET = "enrichment"
API_FEATURE_SETS: tuple[str, ...] = (*CELL_TSORF_FEATURE_SETS, ENRICHMENT_FEATURE_SET)


class GetCellTsorf:
    def __init__(self, feature_store: ParquetFeatureStore) -> None:
        self._feature_store = feature_store

    def execute(
        self,
        region_code: str,
        resolution: int,
        feature_set: str,
        feature: str,
    ) -> list[dict[str, object]]:
        """``[{"hex", "value"}, …]`` for one ЦОФ column across all res cells.

        ``feature_set`` names a Слой 1 set (``walk_dist``, …);
        ``feature`` is one of its columns. Missing partition →
        FileNotFoundError (the API surfaces 404); unknown column →
        KeyError (the API surfaces 400).
        """
        df = self._feature_store.load(region_code, resolution, feature_set)
        if feature not in df.columns:
            available = [c for c in df.columns if c not in {"h3_index", "resolution"}]
            raise KeyError(f"feature {feature!r} not in feature_set={feature_set!r}; available: {available}")
        slim = df.select(["h3_index", pl.col(feature).alias("value")]).drop_nulls("value")
        return [{"hex": r["h3_index"], "value": r["value"]} for r in slim.iter_rows(named=True)]

    def list_features(self, region_code: str, resolution: int, feature_set: str) -> list[str]:
        """Column names of a Слой 1 set, excluding the ``h3_index`` /
        ``resolution`` bookkeeping columns. Empty list when the
        partition is absent so the UI shows an empty selector, not an
        error, for feature sets not yet built in this region."""
        try:
            df = self._feature_store.load(region_code, resolution, feature_set)
        except FileNotFoundError:
            return []
        return [c for c in df.columns if c not in {"h3_index", "resolution"}]

    def feature_set_map(self, region_code: str, resolution: int) -> dict[str, list[str]]:
        """``{feature_set: [features]}`` for every Слой 1 set — one call
        to populate the frontend's two-level (set → feature) selector."""
        return {fs: self.list_features(region_code, resolution, fs) for fs in API_FEATURE_SETS}

    def get_cell_detail(
        self,
        region_code: str,
        resolution: int,
        h3_index: str,
    ) -> dict[str, dict[str, object]]:
        """``{feature_set: {feature_name: value, ...}}`` across all Layer 1 sets for one cell.

        Populates the per-cell inspector when clicking on a hex in «ЦОФ-сетка» mode.
        Missing feature sets are skipped. If the cell is absent across all available
        sets, an empty dictionary is returned.
        """
        detail: dict[str, dict[str, object]] = {}
        for fs in API_FEATURE_SETS:
            try:
                df = self._feature_store.load(region_code, resolution, fs)
            except FileNotFoundError:
                continue
            row = df.filter(pl.col("h3_index") == h3_index)
            if row.is_empty():
                continue
            cols = [c for c in df.columns if c not in {"h3_index", "resolution"}]
            detail[fs] = {c: row[c][0] for c in cols}
        return detail
