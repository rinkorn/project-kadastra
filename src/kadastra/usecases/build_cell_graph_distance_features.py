"""Build Слой 1 walking-distance ЦОФ for point POIs on the cell grid (ADR-0027)."""

from __future__ import annotations

from kadastra.etl.cell_graph_distance_features import (
    compute_cell_graph_distance_features,
)
from kadastra.etl.load_geometries import load_geojsonseq_points
from kadastra.ports.coverage_reader import CoverageReaderPort
from kadastra.ports.feature_store import FeatureStorePort
from kadastra.ports.road_graph import RoadGraphPort


class BuildCellGraphDistanceFeatures:
    def __init__(
        self,
        coverage_reader: CoverageReaderPort,
        feature_store: FeatureStorePort,
        road_graph: RoadGraphPort,
        layer_paths: dict[str, str],
        layer_names: list[str],
    ) -> None:
        self._coverage_reader = coverage_reader
        self._feature_store = feature_store
        self._road_graph = road_graph
        self._layer_paths = layer_paths
        self._layer_names = layer_names

    def execute(self, region_code: str, resolution: int) -> None:
        coverage = self._coverage_reader.load(region_code, resolution)
        point_layers = {name: load_geojsonseq_points(self._layer_paths.get(name, "")) for name in self._layer_names}
        features = compute_cell_graph_distance_features(
            coverage, point_layers=point_layers, road_graph=self._road_graph
        )
        self._feature_store.save(region_code, resolution, "walk_dist", features)
