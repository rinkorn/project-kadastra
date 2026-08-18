"""Build Слой 1 poly-area share ЦОФ on the cell grid (ADR-0027)."""

from __future__ import annotations

from kadastra.etl.cell_polygon_features import compute_cell_polygon_features
from kadastra.etl.load_geometries import load_geojsonseq_geometries
from kadastra.ports.coverage_reader import CoverageReaderPort
from kadastra.ports.feature_store import FeatureStorePort


class BuildCellPolygonFeatures:
    def __init__(
        self,
        coverage_reader: CoverageReaderPort,
        feature_store: FeatureStorePort,
        poly_area_layer_paths: dict[str, str],
        radii_m: list[int],
    ) -> None:
        self._coverage_reader = coverage_reader
        self._feature_store = feature_store
        self._poly_area_layer_paths = poly_area_layer_paths
        self._radii_m = radii_m

    def execute(self, region_code: str, resolution: int) -> None:
        coverage = self._coverage_reader.load(region_code, resolution)
        polygons_by_layer = load_geojsonseq_geometries(self._poly_area_layer_paths)
        features = compute_cell_polygon_features(
            coverage,
            polygons_by_layer=polygons_by_layer,
            radii_m=self._radii_m,
        )
        self._feature_store.save(region_code, resolution, "poly_area", features)
