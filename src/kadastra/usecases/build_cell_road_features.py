"""Build Слой 1 road density ЦОФ on the cell grid (ADR-0027)."""

from __future__ import annotations

import json
from typing import Any, cast

from kadastra.etl.cell_road_features import compute_cell_road_features
from kadastra.ports.coverage_reader import CoverageReaderPort
from kadastra.ports.feature_store import FeatureStorePort
from kadastra.ports.raw_data import RawDataPort


class BuildCellRoadFeatures:
    def __init__(
        self,
        coverage_reader: CoverageReaderPort,
        feature_store: FeatureStorePort,
        raw_data: RawDataPort,
        roads_key: str,
        radius_m: float,
    ) -> None:
        self._coverage_reader = coverage_reader
        self._feature_store = feature_store
        self._raw_data = raw_data
        self._roads_key = roads_key
        self._radius_m = radius_m

    def execute(self, region_code: str, resolution: int) -> None:
        coverage = self._coverage_reader.load(region_code, resolution)
        payload = cast(dict[str, Any], json.loads(self._raw_data.read_bytes(self._roads_key)))
        elements = payload.get("elements", []) or []
        ways = [e for e in elements if e.get("type") == "way" and e.get("geometry")]
        features = compute_cell_road_features(coverage, ways=ways, radius_m=self._radius_m)
        self._feature_store.save(region_code, resolution, "road_density", features)
