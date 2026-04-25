import json
from typing import Any, cast

from kadastra.etl.road_features import compute_road_features
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
        payload = cast(dict[str, Any], json.loads(self._raw_data.read_bytes(self._roads_key)))
        elements = payload.get("elements", []) or []
        ways = [e for e in elements if e.get("type") == "way" and e.get("geometry")]

        for resolution in resolutions:
            coverage = self._coverage_reader.load(region_code, resolution)
            features = compute_road_features(coverage, ways)
            self._feature_store.save(region_code, resolution, "roads", features)
