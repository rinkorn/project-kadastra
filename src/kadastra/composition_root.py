from kadastra.adapters.local_geojson_region_boundary import LocalGeoJsonRegionBoundary
from kadastra.adapters.parquet_coverage_store import ParquetCoverageStore
from kadastra.config import Settings
from kadastra.usecases.build_region_coverage import BuildRegionCoverage


class Container:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def build_region_coverage(self) -> BuildRegionCoverage:
        boundary = LocalGeoJsonRegionBoundary(
            self._settings.region_boundary_path,
            region_code_field=self._settings.region_boundary_field,
        )
        store = ParquetCoverageStore(self._settings.coverage_store_path)
        return BuildRegionCoverage(boundary, store)
