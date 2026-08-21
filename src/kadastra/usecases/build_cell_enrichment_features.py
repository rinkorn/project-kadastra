"""Build the Слой 1 ``enrichment`` feature set (ADR-0029).

Per-cell versions of the ADR-0021..0025 location features that so far
existed only per-object. The same etl functions are reused — they take
a DataFrame with ``lat``/``lon``, so cell centroids pass through the
same code:

- ``dist_to_cbd_m`` — haversine from the cell centre (ADR-0025 п. 1);
- DEM trio — raster sampling at the centre (ADR-0023);
- road-class block — STRtree/UTM-39N from OSM ways (ADR-0024 g. 1);
- ``iso15_*`` — LEFT JOIN of the res-11 isochrone cache (ADR-0024 g. 2);
- heritage ОКН block (ADR-0025 п. 2);
- ЗОУИТ block — spatial join against the silver zones (ADR-0025 п. 3);
- territory columns + ``oktmo_*`` macro — hierarchical H3-mode
  propagation from gold objects + EMISS join (ADR-0029).

Output: ``feature_set=enrichment`` in the Слой 1 feature store, keyed
by ``h3_index``. Each block is opt-in: skipped when its input is not
wired / its silver partition is missing (the ADR-0022/0023 pattern).
"""

from __future__ import annotations

from pathlib import Path

from kadastra.ports.coverage_reader import CoverageReaderPort
from kadastra.ports.dem_sampler import DemSamplerPort
from kadastra.ports.feature_store import FeatureStorePort
from kadastra.ports.valuation_object_reader import ValuationObjectReaderPort

ENRICHMENT_FEATURE_SET = "enrichment"


class BuildCellEnrichmentFeatures:
    def __init__(
        self,
        coverage_reader: CoverageReaderPort,
        feature_store: FeatureStorePort,
        object_reader: ValuationObjectReaderPort,
        *,
        cbd_coords: dict[str, tuple[float, float]],
        dem_sampler: DemSamplerPort | None,
        ways_by_class: dict[str, list[list[tuple[float, float]]]],
        heritage_silver_path: Path | None,
        zouit_silver_path: Path | None,
        isochrone_cache_path: Path | None,
        isochrone_cache_resolution: int,
        macro_oktmo_features_path: Path | None,
        cadastre_target_year: int,
        osm_raions_geojson_path: Path | None = None,
    ) -> None:
        self._coverage_reader = coverage_reader
        self._feature_store = feature_store
        self._object_reader = object_reader
        self._cbd_coords = cbd_coords
        self._dem_sampler = dem_sampler
        self._ways_by_class = ways_by_class
        self._heritage_silver_path = heritage_silver_path
        self._zouit_silver_path = zouit_silver_path
        self._isochrone_cache_path = isochrone_cache_path
        self._isochrone_cache_resolution = isochrone_cache_resolution
        self._macro_oktmo_features_path = macro_oktmo_features_path
        self._cadastre_target_year = cadastre_target_year
        self._osm_raions_geojson_path = osm_raions_geojson_path

    def execute(self, region_code: str, resolution: int) -> None:
        raise NotImplementedError
