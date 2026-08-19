"""Build Слой 1 distance ЦОФ on the cell grid (ADR-0027).

Loads the reference-grid coverage (res 10), computes ``dist_to_<layer>_m``
from each cell centre via ``compute_cell_geom_distance_features``, and
stores the result keyed by ``h3_index`` under feature_set ``geom_distance``.
The object pipeline later joins to this store instead of recomputing
distances per object.
"""

from __future__ import annotations

from kadastra.etl.cell_geom_distance_features import (
    compute_cell_geom_distance_features,
)
from kadastra.etl.load_geometries import load_geojsonseq_geometries
from kadastra.ports.coverage_reader import CoverageReaderPort
from kadastra.ports.feature_store import FeatureStorePort


class BuildCellGeomDistanceFeatures:
    def __init__(
        self,
        coverage_reader: CoverageReaderPort,
        feature_store: FeatureStorePort,
        geom_distance_layer_paths: dict[str, str],
    ) -> None:
        self._coverage_reader = coverage_reader
        self._feature_store = feature_store
        self._geom_distance_layer_paths = geom_distance_layer_paths

    def execute(self, region_code: str, resolution: int) -> None:
        coverage = self._coverage_reader.load(region_code, resolution)
        geometries_by_layer = load_geojsonseq_geometries(self._geom_distance_layer_paths)
        features = compute_cell_geom_distance_features(coverage, geometries_by_layer=geometries_by_layer)
        self._feature_store.save(region_code, resolution, "geom_distance", features)
