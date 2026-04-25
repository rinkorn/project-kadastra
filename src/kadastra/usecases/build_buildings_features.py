from kadastra.ports.coverage_reader import CoverageReaderPort
from kadastra.ports.feature_store import FeatureStorePort
from kadastra.ports.raw_data import RawDataPort


class BuildBuildingsFeatures:
    def __init__(
        self,
        coverage_reader: CoverageReaderPort,
        raw_data: RawDataPort,
        feature_store: FeatureStorePort,
        buildings_key: str,
    ) -> None:
        self._coverage_reader = coverage_reader
        self._raw_data = raw_data
        self._feature_store = feature_store
        self._buildings_key = buildings_key

    def execute(self, region_code: str, resolutions: list[int]) -> None:
        raise NotImplementedError
