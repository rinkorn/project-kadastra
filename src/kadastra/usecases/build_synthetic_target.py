from kadastra.ports.gold_feature_reader import GoldFeatureReaderPort
from kadastra.ports.gold_feature_store import GoldFeatureStorePort


class BuildSyntheticTarget:
    def __init__(
        self,
        gold_reader: GoldFeatureReaderPort,
        target_store: GoldFeatureStorePort,
        seed: int,
    ) -> None:
        raise NotImplementedError

    def execute(self, region_code: str, resolutions: list[int]) -> None:
        raise NotImplementedError
