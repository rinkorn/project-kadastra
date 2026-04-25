from kadastra.ports.gold_feature_reader import GoldFeatureReaderPort


class GetHexFeatures:
    def __init__(self, gold_reader: GoldFeatureReaderPort) -> None:
        self._gold_reader = gold_reader

    def execute(self, region_code: str, resolution: int, feature: str) -> list[dict[str, object]]:
        raise NotImplementedError
