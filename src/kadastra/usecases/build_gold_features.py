from kadastra.ports.coverage_reader import CoverageReaderPort
from kadastra.ports.feature_reader import FeatureReaderPort
from kadastra.ports.gold_feature_store import GoldFeatureStorePort


class BuildGoldFeatures:
    def __init__(
        self,
        coverage_reader: CoverageReaderPort,
        feature_reader: FeatureReaderPort,
        gold_store: GoldFeatureStorePort,
        feature_sets: list[str],
    ) -> None:
        self._coverage_reader = coverage_reader
        self._feature_reader = feature_reader
        self._gold_store = gold_store
        self._feature_sets = feature_sets

    def execute(self, region_code: str, resolutions: list[int]) -> None:
        raise NotImplementedError
