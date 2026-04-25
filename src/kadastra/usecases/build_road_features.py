from kadastra.ports.coverage_reader import CoverageReaderPort
from kadastra.ports.feature_store import FeatureStorePort
from kadastra.ports.raw_data import RawDataPort


class BuildRoadFeatures:
    def __init__(
        self,
        coverage_reader: CoverageReaderPort,
        raw_data: RawDataPort,
        feature_store: FeatureStorePort,
        roads_key: str,
    ) -> None:
        self._coverage_reader = coverage_reader
        self._raw_data = raw_data
        self._feature_store = feature_store
        self._roads_key = roads_key

    def execute(self, region_code: str, resolutions: list[int]) -> None:
        raise NotImplementedError
