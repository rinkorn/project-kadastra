from kadastra.ml.train import CatBoostParams
from kadastra.ports.gold_feature_reader import GoldFeatureReaderPort
from kadastra.ports.model_registry import ModelRegistryPort


class TrainValuationModel:
    def __init__(
        self,
        gold_reader: GoldFeatureReaderPort,
        target_reader: GoldFeatureReaderPort,
        model_registry: ModelRegistryPort,
        params: CatBoostParams,
        n_splits: int,
        parent_resolution: int,
    ) -> None:
        raise NotImplementedError

    def execute(self, region_code: str, resolution: int) -> str:
        raise NotImplementedError
