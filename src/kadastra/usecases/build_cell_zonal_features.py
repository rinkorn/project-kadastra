"""Build Слой 1 zonal density ЦОФ on the cell grid (ADR-0027)."""

from __future__ import annotations

import io

import polars as pl

from kadastra.domain.asset_class import AssetClass
from kadastra.etl.cell_zonal_features import compute_cell_zonal_features
from kadastra.etl.load_geometries import load_geojsonseq_points
from kadastra.ports.coverage_reader import CoverageReaderPort
from kadastra.ports.feature_store import FeatureStorePort
from kadastra.ports.raw_data import RawDataPort
from kadastra.ports.valuation_object_reader import ValuationObjectReaderPort

_CLASS_LAYER_MAP = {
    "apartments": AssetClass.APARTMENT.value,
    "houses": AssetClass.HOUSE.value,
    "commercial": AssetClass.COMMERCIAL.value,
    "landplots": AssetClass.LANDPLOT.value,
}


class BuildCellZonalFeatures:
    def __init__(
        self,
        coverage_reader: CoverageReaderPort,
        feature_store: FeatureStorePort,
        raw_data: RawDataPort,
        object_reader: ValuationObjectReaderPort,
        stations_key: str,
        entrances_key: str,
        radii_m: list[int],
        zonal_layer_names: list[str],
        geom_distance_layer_paths: dict[str, str],
    ) -> None:
        self._coverage_reader = coverage_reader
        self._feature_store = feature_store
        self._raw_data = raw_data
        self._object_reader = object_reader
        self._stations_key = stations_key
        self._entrances_key = entrances_key
        self._radii_m = radii_m
        self._zonal_layer_names = zonal_layer_names
        self._geom_distance_layer_paths = geom_distance_layer_paths

    def execute(self, region_code: str, resolution: int) -> None:
        coverage = self._coverage_reader.load(region_code, resolution)
        stations = pl.read_csv(io.BytesIO(self._raw_data.read_bytes(self._stations_key)))
        entrances = pl.read_csv(io.BytesIO(self._raw_data.read_bytes(self._entrances_key)))
        layers = self._build_layers(region_code, stations, entrances)
        features = compute_cell_zonal_features(coverage, layers=layers, radii_m=self._radii_m)
        self._feature_store.save(region_code, resolution, "zonal", features)

    def _build_layers(
        self,
        region_code: str,
        stations: pl.DataFrame,
        entrances: pl.DataFrame,
    ) -> dict[str, pl.DataFrame]:
        layers: dict[str, pl.DataFrame] = {}
        for name in self._zonal_layer_names:
            if name == "stations":
                layers[name] = stations.select(["lat", "lon"])
            elif name == "entrances":
                layers[name] = entrances.select(["lat", "lon"])
            elif name in _CLASS_LAYER_MAP:
                objs = self._object_reader.load(region_code, AssetClass(_CLASS_LAYER_MAP[name]))
                layers[name] = objs.select(["lat", "lon"])
            elif name in self._geom_distance_layer_paths:
                layers[name] = load_geojsonseq_points(self._geom_distance_layer_paths[name])
        return layers
