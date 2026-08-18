"""Build Слой 1 metro accessibility ЦОФ on the cell grid (ADR-0027)."""

from __future__ import annotations

import io

import polars as pl

from kadastra.etl.cell_metro_features import compute_cell_metro_features
from kadastra.ports.coverage_reader import CoverageReaderPort
from kadastra.ports.feature_store import FeatureStorePort
from kadastra.ports.raw_data import RawDataPort
from kadastra.ports.road_graph import RoadGraphPort


class BuildCellMetroFeatures:
    def __init__(
        self,
        coverage_reader: CoverageReaderPort,
        feature_store: FeatureStorePort,
        raw_data: RawDataPort,
        road_graph: RoadGraphPort,
        stations_key: str,
        entrances_key: str,
    ) -> None:
        self._coverage_reader = coverage_reader
        self._feature_store = feature_store
        self._raw_data = raw_data
        self._road_graph = road_graph
        self._stations_key = stations_key
        self._entrances_key = entrances_key

    def execute(self, region_code: str, resolution: int) -> None:
        coverage = self._coverage_reader.load(region_code, resolution)
        stations = pl.read_csv(io.BytesIO(self._raw_data.read_bytes(self._stations_key)))
        entrances = pl.read_csv(io.BytesIO(self._raw_data.read_bytes(self._entrances_key)))
        features = compute_cell_metro_features(coverage, stations, entrances, road_graph=self._road_graph)
        self._feature_store.save(region_code, resolution, "metro", features)
