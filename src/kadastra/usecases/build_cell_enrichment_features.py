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

import polars as pl

from kadastra.domain.asset_class import AssetClass
from kadastra.etl.cell_territory_features import compute_cell_territory_features
from kadastra.etl.h3_coverage import h3_cells_to_latlng
from kadastra.etl.load_geometries import load_named_geojsonseq_polygons
from kadastra.etl.object_cbd_distance import compute_cbd_distance
from kadastra.etl.object_dem_features import compute_object_dem_features
from kadastra.etl.object_heritage_features import compute_object_heritage_features
from kadastra.etl.object_isochrone_features import join_isochrone_features
from kadastra.etl.object_macro_features import compute_object_macro_features
from kadastra.etl.object_road_class_features import compute_nearest_road_features
from kadastra.etl.object_zouit_features import compute_object_zouit_features
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
        coverage = self._coverage_reader.load(region_code, resolution)
        centers = h3_cells_to_latlng(coverage["h3_index"].to_list())
        # Pseudo-object frame: object_id = h3_index so the per-object etl
        # helpers (keyed by object_id) run unchanged on cell centroids.
        # The h3_index column is deliberately absent — some helpers
        # (isochrone join) manage their own temp h3_index and drop it.
        cells = pl.DataFrame(
            {
                "object_id": coverage["h3_index"],
                "lat": pl.Series([c[0] for c in centers], dtype=pl.Float64),
                "lon": pl.Series([c[1] for c in centers], dtype=pl.Float64),
            }
        )

        enriched = cells
        if region_code in self._cbd_coords:
            cbd_lat, cbd_lon = self._cbd_coords[region_code]
            enriched = compute_cbd_distance(enriched, cbd_lat=cbd_lat, cbd_lon=cbd_lon)
        if self._dem_sampler is not None:
            enriched = compute_object_dem_features(enriched, dem_sampler=self._dem_sampler)
        if self._ways_by_class:
            road = compute_nearest_road_features(
                enriched.select(["object_id", "lat", "lon"]),
                ways_by_class=self._ways_by_class,
            )
            enriched = enriched.join(road, on="object_id", how="left")
        if self._isochrone_cache_path is not None:
            iso_cache = self._load_partition(
                self._isochrone_cache_path
                / f"region={region_code}"
                / f"h3_p={self._isochrone_cache_resolution}"
                / "data.parquet"
            )
            if iso_cache is not None:
                enriched = join_isochrone_features(enriched, iso_cache, resolution=self._isochrone_cache_resolution)
        if self._heritage_silver_path is not None:
            heritage = self._load_partition(self._heritage_silver_path / f"region={region_code}" / "data.parquet")
            if heritage is not None:
                enriched = compute_object_heritage_features(enriched, heritage=heritage)
        if self._zouit_silver_path is not None:
            zones = self._load_partition(self._zouit_silver_path / f"region={region_code}" / "data.parquet")
            if zones is not None:
                zouit = compute_object_zouit_features(enriched.select(["object_id", "lat", "lon"]), zones=zones)
                enriched = enriched.join(zouit, on="object_id", how="left")
        # Territory propagation + macro-OKTMO join (ADR-0029). The
        # isochrone join above drops the h3_index helper column, so the
        # territory frame is keyed back via object_id.
        objects = self._load_all_objects(region_code)
        territory = compute_cell_territory_features(
            enriched.select(pl.col("object_id").alias("h3_index"), "lat", "lon"),
            objects,
            raion_polygons=load_named_geojsonseq_polygons(self._osm_raions_geojson_path),
            cell_resolution=resolution,
        )
        enriched = enriched.join(territory, left_on="object_id", right_on="h3_index", how="left")
        if self._macro_oktmo_features_path is not None:
            macro_table = self._load_macro_oktmo_features(region_code)
            if macro_table is not None:
                enriched = compute_object_macro_features(
                    enriched,
                    macro_table=macro_table,
                    target_year=self._cadastre_target_year,
                )

        out = (
            enriched.with_columns(pl.col("object_id").alias("h3_index"))
            .drop(["object_id", "lat", "lon"])
            .with_columns(pl.lit(resolution, dtype=pl.Int64).alias("resolution"))
        )
        self._feature_store.save(region_code, resolution, ENRICHMENT_FEATURE_SET, out)

    def _load_all_objects(self, region_code: str) -> pl.DataFrame:
        """Combined gold objects of all classes (territory propagation source)."""
        frames = []
        for ac in AssetClass:
            df = self._object_reader.load(region_code, ac)
            if not df.is_empty():
                frames.append(df)
        return pl.concat(frames, how="vertical_relaxed") if frames else pl.DataFrame()

    @staticmethod
    def _load_partition(path: Path) -> pl.DataFrame | None:
        return pl.read_parquet(path) if path.is_file() else None

    def _load_macro_oktmo_features(self, region_code: str) -> pl.DataFrame | None:
        """Wide per-(oktmo, year) EMISS macro table (ADR-0022 layout)."""
        base = self._macro_oktmo_features_path
        if base is None:
            return None
        paths = sorted(base.glob(f"region={region_code}/year=*/data.parquet"))
        if not paths:
            return None
        return pl.concat([pl.read_parquet(p) for p in paths])
