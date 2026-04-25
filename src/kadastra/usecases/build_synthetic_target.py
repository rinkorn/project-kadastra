from kadastra.etl.synthetic_target import compute_synthetic_target
from kadastra.ports.gold_feature_reader import GoldFeatureReaderPort
from kadastra.ports.gold_feature_store import GoldFeatureStorePort


class BuildSyntheticTarget:
    def __init__(
        self,
        gold_reader: GoldFeatureReaderPort,
        target_store: GoldFeatureStorePort,
        seed: int,
    ) -> None:
        self._gold_reader = gold_reader
        self._target_store = target_store
        self._seed = seed

    def execute(self, region_code: str, resolutions: list[int]) -> None:
        for resolution in resolutions:
            gold = self._gold_reader.load(region_code, resolution)
            df = compute_synthetic_target(gold, seed=self._seed)
            self._target_store.save(region_code, resolution, df)
