from pathlib import Path

from fastapi import FastAPI

from kadastra.adapters.local_ebm_model_loader import LocalEbmModelLoader
from kadastra.adapters.local_geojson_region_boundary import LocalGeoJsonRegionBoundary
from kadastra.adapters.local_model_loader import LocalModelLoader
from kadastra.adapters.local_model_registry import LocalModelRegistry
from kadastra.adapters.local_oof_predictions_reader import LocalOofPredictionsReader
from kadastra.adapters.mlflow_model_loader import MLflowModelLoader
from kadastra.adapters.mlflow_model_registry import MLflowModelRegistry
from kadastra.adapters.networkx_road_graph import NetworkxRoadGraph
from kadastra.adapters.parquet_coverage_store import ParquetCoverageStore
from kadastra.adapters.parquet_feature_store import ParquetFeatureStore
from kadastra.adapters.parquet_nspd_silver_store import ParquetNspdSilverStore
from kadastra.adapters.parquet_valuation_object_store import ParquetValuationObjectStore
from kadastra.adapters.rasterio_dem_sampler import RasterioDemSampler
from kadastra.adapters.s3_raw_data import S3RawData
from kadastra.api.auth import BearerAuthMiddleware
from kadastra.api.routes import make_api_router
from kadastra.config import Settings
from kadastra.etl.object_road_class_features import (
    parse_major_road_ways,
    parse_minor_road_ways,
)
from kadastra.ml.train import CatBoostParams
from kadastra.ports.model_loader import ModelLoaderPort
from kadastra.ports.model_registry import ModelRegistryPort
from kadastra.ports.road_graph import RoadGraphPort
from kadastra.usecases.assemble_nspd_valuation_objects import (
    AssembleNspdValuationObjects,
)
from kadastra.usecases.build_cell_enrichment_features import BuildCellEnrichmentFeatures
from kadastra.usecases.build_cell_geom_distance_features import (
    BuildCellGeomDistanceFeatures,
)
from kadastra.usecases.build_cell_graph_distance_features import (
    BuildCellGraphDistanceFeatures,
)
from kadastra.usecases.build_cell_metro_features import (
    BuildCellMetroFeatures,
)
from kadastra.usecases.build_cell_polygon_features import (
    BuildCellPolygonFeatures,
)
from kadastra.usecases.build_cell_road_features import (
    BuildCellRoadFeatures,
)
from kadastra.usecases.build_cell_valuation import (
    CELL_VALUATION_FEATURE_SETS,
    BuildCellValuation,
)
from kadastra.usecases.build_cell_zonal_features import (
    BuildCellZonalFeatures,
)
from kadastra.usecases.build_dem_silver import BuildDemSilver
from kadastra.usecases.build_hex_aggregates import BuildHexAggregates
from kadastra.usecases.build_object_features import BuildObjectFeatures
from kadastra.usecases.build_object_synthetic_target import BuildObjectSyntheticTarget
from kadastra.usecases.build_region_coverage import BuildRegionCoverage
from kadastra.usecases.build_representativeness_report import BuildRepresentativenessReport
from kadastra.usecases.build_valuation_objects import BuildValuationObjects
from kadastra.usecases.get_cell_tsorf import GetCellTsorf
from kadastra.usecases.get_hex_aggregates import GetHexAggregates
from kadastra.usecases.get_market_reference import GetMarketReference
from kadastra.usecases.infer_object_valuation import InferObjectValuation
from kadastra.usecases.load_nspd_raw_objects import LoadNspdRawObjects
from kadastra.usecases.load_object_inspection import LoadObjectInspection
from kadastra.usecases.train_object_valuation_model import TrainObjectValuationModel
from kadastra.usecases.train_quartet import TrainQuartet
from kadastra.web.routes import make_web_router

_OBJECT_RUN_NAME_PREFIX = "quartet-object-"  # ADR-0016 quartet runs; primary artifact is the CatBoost model.cbm


class Container:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def build_region_coverage(self) -> BuildRegionCoverage:
        boundary = LocalGeoJsonRegionBoundary(
            self._settings.region_boundary_path,
            region_code_field=self._settings.region_boundary_field,
        )
        store = ParquetCoverageStore(self._settings.coverage_store_path)
        return BuildRegionCoverage(boundary, store)

    def build_s3_raw_data(self) -> S3RawData:
        s = self._settings
        if not (s.s3_bucket and s.s3_access_key and s.s3_secret_key):
            raise RuntimeError("S3 credentials not configured: set S3_BUCKET, S3_ACCESS_KEY, S3_SECRET_KEY in .env")
        return S3RawData(
            bucket=s.s3_bucket,
            access_key=s.s3_access_key,
            secret_key=s.s3_secret_key,
            endpoint_url=s.s3_endpoint_url,
            region=s.s3_region,
            addressing_style=s.s3_addressing_style,
        )

    def build_cell_geom_distance_features(self) -> BuildCellGeomDistanceFeatures:
        s = self._settings
        return BuildCellGeomDistanceFeatures(
            coverage_reader=ParquetCoverageStore(s.coverage_store_path),
            feature_store=ParquetFeatureStore(s.feature_store_path),
            geom_distance_layer_paths=s.geom_distance_layer_paths,
        )

    def build_cell_graph_distance_features(self) -> BuildCellGraphDistanceFeatures:
        s = self._settings
        return BuildCellGraphDistanceFeatures(
            coverage_reader=ParquetCoverageStore(s.coverage_store_path),
            feature_store=ParquetFeatureStore(s.feature_store_path),
            road_graph=self.build_road_graph(),
            layer_paths=s.geom_distance_layer_paths,
            layer_names=s.walk_dist_layer_names,
        )

    def build_cell_metro_features(self) -> BuildCellMetroFeatures:
        s = self._settings
        return BuildCellMetroFeatures(
            coverage_reader=ParquetCoverageStore(s.coverage_store_path),
            feature_store=ParquetFeatureStore(s.feature_store_path),
            raw_data=self.build_s3_raw_data(),
            road_graph=self.build_road_graph(),
            stations_key=s.metro_stations_key,
            entrances_key=s.metro_entrances_key,
        )

    def build_cell_polygon_features(self) -> BuildCellPolygonFeatures:
        s = self._settings
        return BuildCellPolygonFeatures(
            coverage_reader=ParquetCoverageStore(s.coverage_store_path),
            feature_store=ParquetFeatureStore(s.feature_store_path),
            poly_area_layer_paths=s.poly_area_layer_paths,
            radii_m=s.poly_area_radii_m,
        )

    def build_cell_zonal_features(self) -> BuildCellZonalFeatures:
        s = self._settings
        return BuildCellZonalFeatures(
            coverage_reader=ParquetCoverageStore(s.coverage_store_path),
            feature_store=ParquetFeatureStore(s.feature_store_path),
            raw_data=self.build_s3_raw_data(),
            object_reader=ParquetValuationObjectStore(s.valuation_object_store_path),
            stations_key=s.metro_stations_key,
            entrances_key=s.metro_entrances_key,
            radii_m=s.zonal_radii_m,
            zonal_layer_names=s.zonal_layer_names,
            geom_distance_layer_paths=s.geom_distance_layer_paths,
        )

    def build_cell_road_features(self) -> BuildCellRoadFeatures:
        s = self._settings
        return BuildCellRoadFeatures(
            coverage_reader=ParquetCoverageStore(s.coverage_store_path),
            feature_store=ParquetFeatureStore(s.feature_store_path),
            raw_data=self.build_s3_raw_data(),
            roads_key=s.roads_key,
            radius_m=s.object_road_radius_m,
        )

    def build_model_registry(self) -> ModelRegistryPort:
        s = self._settings
        if s.mlflow_enabled:
            if not s.mlflow_tracking_uri:
                raise RuntimeError("MLFLOW_TRACKING_URI is required when MLFLOW_ENABLED=True")
            return MLflowModelRegistry(
                tracking_uri=s.mlflow_tracking_uri,
                experiment_name=s.mlflow_experiment_name,
            )
        return LocalModelRegistry(s.model_registry_path)

    def build_model_loader(self) -> ModelLoaderPort:
        s = self._settings
        if s.mlflow_enabled:
            if not s.mlflow_tracking_uri:
                raise RuntimeError("MLFLOW_TRACKING_URI is required when MLFLOW_ENABLED=True")
            return MLflowModelLoader(
                tracking_uri=s.mlflow_tracking_uri,
                experiment_name=s.mlflow_experiment_name,
            )
        return LocalModelLoader(s.model_registry_path)

    def build_valuation_objects(self) -> BuildValuationObjects:
        s = self._settings
        return BuildValuationObjects(
            raw_data=self.build_s3_raw_data(),
            store=ParquetValuationObjectStore(s.valuation_object_store_path),
            buildings_key=s.buildings_key,
        )

    def build_load_nspd_raw_objects(self) -> LoadNspdRawObjects:
        s = self._settings
        return LoadNspdRawObjects(
            region_boundary=LocalGeoJsonRegionBoundary(
                s.region_boundary_path,
                region_code_field=s.region_boundary_field,
            ),
            silver_store=ParquetNspdSilverStore(s.nspd_silver_store_path),
        )

    def build_assemble_nspd_valuation_objects(self) -> AssembleNspdValuationObjects:
        s = self._settings
        return AssembleNspdValuationObjects(
            silver_store=ParquetNspdSilverStore(s.nspd_silver_store_path),
            valuation_object_store=ParquetValuationObjectStore(s.valuation_object_store_path),
        )

    def build_road_graph(self) -> RoadGraphPort:
        return NetworkxRoadGraph.from_parquet(self._settings.road_graph_edges_path)

    def build_dem_silver(self) -> BuildDemSilver:
        s = self._settings
        return BuildDemSilver(
            dem_raw_dir=s.dem_raw_dir,
            output_base_path=s.dem_silver_base_path,
            relief_radius_m=s.dem_relief_radius_m,
        )

    def build_dem_sampler(self) -> RasterioDemSampler | None:
        """ADR-0023: sampler over the silver DEM rasters for the region.

        Returns None (the BuildObjectFeatures DEM step is skipped) when
        the silver layers were not built for the region yet — mirrors
        the gar-lookup / macro-oktmo opt-in pattern.
        """
        s = self._settings
        base = s.dem_silver_base_path / f"region={s.region_code}"
        elevation = base / "elevation.tif"
        slope = base / "slope_deg.tif"
        relief = base / "relative_relief_500m.tif"
        if not (elevation.is_file() and slope.is_file() and relief.is_file()):
            return None
        return RasterioDemSampler(elevation_path=elevation, slope_path=slope, relief_path=relief)

    def build_object_features(self) -> BuildObjectFeatures:
        s = self._settings
        store = ParquetValuationObjectStore(s.valuation_object_store_path)
        return BuildObjectFeatures(
            reader=store,
            store=store,
            raw_data=self.build_s3_raw_data(),
            stations_key=s.metro_stations_key,
            entrances_key=s.metro_entrances_key,
            roads_key=s.roads_key,
            neighbor_radius_m=s.object_neighbor_radius_m,
            road_radius_m=s.object_road_radius_m,
            road_graph=(self.build_road_graph() if not s.cell_tsorf_enabled else None),
            relative_feature_parent_resolutions=s.relative_feature_parent_resolutions,
            relative_feature_columns=s.relative_feature_columns,
            zonal_radii_m=s.zonal_radii_m,
            zonal_layer_names=s.zonal_layer_names,
            poly_area_radii_m=s.poly_area_radii_m,
            poly_area_layer_paths=s.poly_area_layer_paths,
            geom_distance_layer_paths=s.geom_distance_layer_paths,
            gar_lookup_cadnum_index_path=s.gar_lookup_cadnum_index_path,
            gar_lookup_mun_lookup_path=s.gar_lookup_mun_lookup_path,
            gar_lookup_object_params_path=s.gar_lookup_object_params_path,
            osm_raions_geojson_path=s.osm_raions_geojson_path,
            current_year_for_age_features=s.current_year_for_age_features,
            cell_geom_distance_reader=(ParquetFeatureStore(s.feature_store_path) if s.cell_tsorf_enabled else None),
            cell_polygon_reader=(ParquetFeatureStore(s.feature_store_path) if s.cell_tsorf_enabled else None),
            cell_zonal_reader=(ParquetFeatureStore(s.feature_store_path) if s.cell_tsorf_enabled else None),
            cell_road_reader=(ParquetFeatureStore(s.feature_store_path) if s.cell_tsorf_enabled else None),
            cell_metro_reader=(ParquetFeatureStore(s.feature_store_path) if s.cell_tsorf_enabled else None),
            cell_walk_dist_reader=(ParquetFeatureStore(s.feature_store_path) if s.cell_tsorf_enabled else None),
            cell_tsorf_resolution=s.cell_tsorf_resolution,
            cell_tsorf_overlap_weighted=s.cell_tsorf_overlap_weighted,
            macro_oktmo_features_path=(s.macro_oktmo_features_path if s.macro_emiss_enabled else None),
            cadastre_target_year=s.cadastre_target_year,
            dem_sampler=(self.build_dem_sampler() if s.dem_features_enabled else None),
            road_class_features_path=s.road_class_features_path,
            isochrone_cache_path=s.isochrone_cache_path,
            isochrone_cache_resolution=s.isochrone_cache_resolution,
            cbd_coords=s.cbd_coords,
            heritage_silver_path=s.heritage_silver_path,
            zouit_features_path=s.zouit_features_path,
        )

    def build_object_synthetic_target(self) -> BuildObjectSyntheticTarget:
        s = self._settings
        store = ParquetValuationObjectStore(s.valuation_object_store_path)
        return BuildObjectSyntheticTarget(
            reader=store,
            store=store,
            seed=s.synthetic_target_seed,
        )

    def build_train_object_valuation_model(self) -> TrainObjectValuationModel:
        s = self._settings
        params = CatBoostParams(
            iterations=s.catboost_iterations,
            learning_rate=s.catboost_learning_rate,
            depth=s.catboost_depth,
            seed=s.catboost_seed,
        )
        return TrainObjectValuationModel(
            reader=ParquetValuationObjectStore(s.valuation_object_store_path),
            model_registry=self.build_model_registry(),
            params=params,
            n_splits=s.train_n_splits,
            parent_resolution=s.train_parent_resolution,
        )

    def build_train_quartet(self) -> TrainQuartet:
        s = self._settings
        params = CatBoostParams(
            iterations=s.catboost_iterations,
            learning_rate=s.catboost_learning_rate,
            depth=s.catboost_depth,
            seed=s.catboost_seed,
        )
        return TrainQuartet(
            reader=ParquetValuationObjectStore(s.valuation_object_store_path),
            model_registry=self.build_model_registry(),
            catboost_params=params,
            ebm_max_bins=s.ebm_max_bins,
            ebm_interactions=s.ebm_interactions,
            grey_tree_max_depth=s.grey_tree_max_depth,
            n_splits=s.train_n_splits,
            parent_resolution=s.train_parent_resolution,
            parallel_folds=s.quartet_parallel_folds,
            skip_final_simplifier_fits=s.quartet_skip_final_simplifier_fits,
        )

    def build_hex_aggregates(self) -> BuildHexAggregates:
        s = self._settings
        return BuildHexAggregates(
            reader=ParquetValuationObjectStore(s.valuation_object_store_path),
            oof_reader=LocalOofPredictionsReader(s.model_registry_path),
            output_base_path=s.hex_aggregates_base_path,
            resolutions=s.hex_aggregates_resolutions,
        )

    def build_infer_object_valuation(self) -> InferObjectValuation:
        s = self._settings
        return InferObjectValuation(
            model_loader=self.build_model_loader(),
            reader=ParquetValuationObjectStore(s.valuation_object_store_path),
            prediction_store=ParquetValuationObjectStore(s.object_predictions_store_path),
            run_name_prefix=_OBJECT_RUN_NAME_PREFIX,
        )

    def build_get_hex_aggregates(self) -> GetHexAggregates:
        s = self._settings
        return GetHexAggregates(s.hex_aggregates_base_path)

    def build_get_market_reference(self) -> GetMarketReference:
        s = self._settings
        return GetMarketReference(s.emiss_silver_base_path)

    def build_load_object_inspection(self) -> LoadObjectInspection:
        s = self._settings
        return LoadObjectInspection(
            reader=ParquetValuationObjectStore(s.valuation_object_store_path),
            oof_reader=LocalOofPredictionsReader(s.model_registry_path),
            ebm_loader=LocalEbmModelLoader(s.model_registry_path),
        )

    def build_get_cell_tsorf(self) -> GetCellTsorf:
        s = self._settings
        return GetCellTsorf(ParquetFeatureStore(s.feature_store_path))

    def build_representativeness_report(self) -> BuildRepresentativenessReport:
        s = self._settings
        return BuildRepresentativenessReport(
            feature_store=ParquetFeatureStore(s.feature_store_path),
            object_reader=ParquetValuationObjectStore(s.valuation_object_store_path),
            output_base_path=s.representativeness_path,
            resolution=s.cell_tsorf_resolution,
        )

    def build_cell_enrichment_features(self) -> BuildCellEnrichmentFeatures:
        s = self._settings
        raw_data = self.build_s3_raw_data()
        ways_by_class = parse_major_road_ways(raw_data.read_bytes(s.roads_key))
        for cls, ways in parse_minor_road_ways(s.minor_road_ways_path).items():
            ways_by_class.setdefault(cls, []).extend(ways)
        return BuildCellEnrichmentFeatures(
            coverage_reader=ParquetCoverageStore(s.coverage_store_path),
            feature_store=ParquetFeatureStore(s.feature_store_path),
            object_reader=ParquetValuationObjectStore(s.valuation_object_store_path),
            cbd_coords=s.cbd_coords,
            dem_sampler=(self.build_dem_sampler() if s.dem_features_enabled else None),
            ways_by_class=ways_by_class,
            heritage_silver_path=s.heritage_silver_path,
            zouit_silver_path=s.zouit_silver_path,
            isochrone_cache_path=s.isochrone_cache_path,
            isochrone_cache_resolution=s.isochrone_cache_resolution,
            macro_oktmo_features_path=(s.macro_oktmo_features_path if s.macro_emiss_enabled else None),
            cadastre_target_year=s.cadastre_target_year,
            osm_raions_geojson_path=s.osm_raions_geojson_path,
        )

    def build_cell_valuation(self) -> BuildCellValuation:
        s = self._settings
        return BuildCellValuation(
            cell_feature_reader=ParquetFeatureStore(s.feature_store_path),
            object_reader=ParquetValuationObjectStore(s.valuation_object_store_path),
            ebm_loader=LocalEbmModelLoader(s.model_registry_path),
            output_store=ParquetValuationObjectStore(s.cell_valuation_store_path),
            cell_feature_sets=CELL_VALUATION_FEATURE_SETS,
            resolution=s.cell_tsorf_resolution,
            relative_parent_resolutions=s.relative_feature_parent_resolutions,
            relative_feature_columns=s.relative_feature_columns,
            current_year=s.current_year_for_age_features,
            landplot_vri_top_n=s.landplot_vri_top_n,
        )


def create_app(settings: Settings) -> FastAPI:
    container = Container(settings)
    templates_dir = Path(__file__).parent / "web" / "templates"

    app = FastAPI(title="kadastra")

    if settings.auth_token:
        app.add_middleware(BearerAuthMiddleware, token=settings.auth_token)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(
        make_api_router(
            region_code=settings.region_code,
            get_hex_aggregates=container.build_get_hex_aggregates(),
            load_inspection=container.build_load_object_inspection(),
            get_market_reference=container.build_get_market_reference(),
            market_reference_year=settings.emiss_market_reference_year,
            get_cell_tsorf=container.build_get_cell_tsorf(),
        )
    )
    app.include_router(make_web_router(templates_dir))
    return app
