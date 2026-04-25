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
        for resolution in resolutions:
            df = self._coverage_reader.load(region_code, resolution)
            for feature_set in self._feature_sets:
                features = self._feature_reader.load(region_code, resolution, feature_set)
                df = df.join(features, on=["h3_index", "resolution"], how="left")
            self._gold_store.save(region_code, resolution, df)
