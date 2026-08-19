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

# The six Слой 1 feature sets built by BuildCell*Features. Drives the
# frontend's feature_set selector. Listed here (not introspected from
# the store) so a missing partition reads as an empty set rather than
# silently disappearing from the UI.
CELL_TSORF_FEATURE_SETS: tuple[str, ...] = (
    "geom_distance",
    "poly_area",
    "zonal",
    "road_density",
    "metro",
    "walk_dist",
)


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
        return {fs: self.list_features(region_code, resolution, fs) for fs in CELL_TSORF_FEATURE_SETS}
