from kadastra.etl.h3_coverage import geometry_to_h3_cells
from kadastra.ports.coverage_store import CoverageStorePort
from kadastra.ports.region_boundary import RegionBoundaryPort


class BuildRegionCoverage:
    def __init__(self, boundary: RegionBoundaryPort, store: CoverageStorePort) -> None:
        self._boundary = boundary
        self._store = store

    def execute(self, region_code: str, resolutions: list[int]) -> None:
        geometry = self._boundary.get_boundary(region_code)
        cells: list[tuple[str, int]] = [
            (cell, resolution)
            for resolution in resolutions
            for cell in geometry_to_h3_cells(geometry, resolution)
        ]
        self._store.save(region_code, cells)
