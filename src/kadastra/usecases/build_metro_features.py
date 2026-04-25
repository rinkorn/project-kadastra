import io

import polars as pl

from kadastra.etl.metro_features import compute_metro_features
from kadastra.ports.coverage_reader import CoverageReaderPort
from kadastra.ports.feature_store import FeatureStorePort
from kadastra.ports.raw_data import RawDataPort


class BuildMetroFeatures:
    def __init__(
        self,
        coverage_reader: CoverageReaderPort,
        raw_data: RawDataPort,
        feature_store: FeatureStorePort,
        stations_key: str,
        entrances_key: str,
    ) -> None:
        self._coverage_reader = coverage_reader
        self._raw_data = raw_data
        self._feature_store = feature_store
        self._stations_key = stations_key
        self._entrances_key = entrances_key

    def execute(self, region_code: str, resolutions: list[int]) -> None:
        stations = pl.read_csv(io.BytesIO(self._raw_data.read_bytes(self._stations_key)))
        entrances = pl.read_csv(io.BytesIO(self._raw_data.read_bytes(self._entrances_key)))

        for resolution in resolutions:
            coverage = self._coverage_reader.load(region_code, resolution)
            features = compute_metro_features(coverage, stations, entrances)
            self._feature_store.save(region_code, resolution, "metro", features)
