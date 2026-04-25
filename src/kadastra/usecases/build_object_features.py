import io
import json
from typing import Any, cast

import polars as pl

from kadastra.domain.asset_class import AssetClass
from kadastra.etl.object_metro_features import compute_object_metro_features
from kadastra.etl.object_neighbor_features import compute_object_neighbor_features
from kadastra.etl.object_road_features import compute_object_road_features
from kadastra.ports.raw_data import RawDataPort
from kadastra.ports.road_graph import RoadGraphPort
from kadastra.ports.valuation_object_reader import ValuationObjectReaderPort
from kadastra.ports.valuation_object_store import ValuationObjectStorePort


class BuildObjectFeatures:
    def __init__(
        self,
        reader: ValuationObjectReaderPort,
        store: ValuationObjectStorePort,
        raw_data: RawDataPort,
        stations_key: str,
        entrances_key: str,
        roads_key: str,
        neighbor_radius_m: float,
        road_radius_m: float,
        road_graph: RoadGraphPort,
    ) -> None:
        self._reader = reader
        self._store = store
        self._raw_data = raw_data
        self._stations_key = stations_key
        self._entrances_key = entrances_key
        self._roads_key = roads_key
        self._neighbor_radius_m = neighbor_radius_m
        self._road_radius_m = road_radius_m
        self._road_graph = road_graph

    def execute(self, region_code: str, asset_classes: list[AssetClass]) -> None:
        stations = pl.read_csv(
            io.BytesIO(self._raw_data.read_bytes(self._stations_key))
        )
        entrances = pl.read_csv(
            io.BytesIO(self._raw_data.read_bytes(self._entrances_key))
        )

        roads_payload = cast(
            dict[str, Any], json.loads(self._raw_data.read_bytes(self._roads_key))
        )
        elements = roads_payload.get("elements", []) or []
        ways = [e for e in elements if e.get("type") == "way" and e.get("geometry")]

        slices = {ac: self._reader.load(region_code, ac) for ac in asset_classes}
        non_empty = [df for df in slices.values() if not df.is_empty()]
        combined = (
            pl.concat(non_empty, how="vertical_relaxed")
            if non_empty
            else next(iter(slices.values()))
        )

        enriched = compute_object_metro_features(
            combined, stations, entrances, road_graph=self._road_graph
        )
        enriched = compute_object_road_features(
            enriched, ways, radius_m=self._road_radius_m
        )
        enriched = compute_object_neighbor_features(
            enriched, radius_m=self._neighbor_radius_m
        )

        for asset_class in asset_classes:
            slice_df = enriched.filter(pl.col("asset_class") == asset_class.value)
            self._store.save(region_code, asset_class, slice_df)
