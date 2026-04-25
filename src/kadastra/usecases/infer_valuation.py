from kadastra.ports.gold_feature_reader import GoldFeatureReaderPort
from kadastra.ports.gold_feature_store import GoldFeatureStorePort
from kadastra.ports.model_loader import ModelLoaderPort


class InferValuation:
    def __init__(
        self,
        model_loader: ModelLoaderPort,
        gold_reader: GoldFeatureReaderPort,
        prediction_store: GoldFeatureStorePort,
        run_name_prefix: str,
    ) -> None:
        raise NotImplementedError

    def execute(self, region_code: str, resolution: int, *, run_id: str | None = None) -> str:
        raise NotImplementedError
