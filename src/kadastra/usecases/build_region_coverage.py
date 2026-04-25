from kadastra.ports.coverage_store import CoverageStorePort
from kadastra.ports.region_boundary import RegionBoundaryPort


class BuildRegionCoverage:
    def __init__(self, boundary: RegionBoundaryPort, store: CoverageStorePort) -> None:
        self._boundary = boundary
        self._store = store

    def execute(self, region_code: str, resolutions: list[int]) -> None:
        raise NotImplementedError
