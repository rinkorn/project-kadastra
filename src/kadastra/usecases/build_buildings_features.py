import io

import polars as pl

from kadastra.etl.building_features import compute_building_features
from kadastra.ports.coverage_reader import CoverageReaderPort
from kadastra.ports.feature_store import FeatureStorePort
from kadastra.ports.raw_data import RawDataPort

# OSM building CSVs have ambiguous columns: housenumber mixes digits and letters,
# levels may be "2-3", start_date may be a year or a full date. Read everything
# potentially-mixed as Utf8; compute_building_features casts levels/flats with strict=False.
_BUILDINGS_SCHEMA_OVERRIDES = {
    "levels": pl.Utf8,
    "flats": pl.Utf8,
    "housenumber": pl.Utf8,
    "postcode": pl.Utf8,
    "start_date": pl.Utf8,
}


class BuildBuildingsFeatures:
    def __init__(
        self,
        coverage_reader: CoverageReaderPort,
        raw_data: RawDataPort,
        feature_store: FeatureStorePort,
        buildings_key: str,
    ) -> None:
        self._coverage_reader = coverage_reader
        self._raw_data = raw_data
        self._feature_store = feature_store
        self._buildings_key = buildings_key

    def execute(self, region_code: str, resolutions: list[int]) -> None:
        buildings = pl.read_csv(
            io.BytesIO(self._raw_data.read_bytes(self._buildings_key)),
            schema_overrides=_BUILDINGS_SCHEMA_OVERRIDES,
        )

        for resolution in resolutions:
            coverage = self._coverage_reader.load(region_code, resolution)
            features = compute_building_features(coverage, buildings)
            self._feature_store.save(region_code, resolution, "buildings", features)
