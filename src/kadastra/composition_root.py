from pathlib import Path

from fastapi import FastAPI

from kadastra.adapters.local_geojson_region_boundary import LocalGeoJsonRegionBoundary
from kadastra.adapters.local_model_registry import LocalModelRegistry
from kadastra.adapters.mlflow_model_registry import MLflowModelRegistry
from kadastra.adapters.parquet_coverage_store import ParquetCoverageStore
from kadastra.adapters.parquet_feature_store import ParquetFeatureStore
from kadastra.adapters.parquet_gold_feature_store import ParquetGoldFeatureStore
from kadastra.adapters.s3_raw_data import S3RawData
from kadastra.api.routes import make_api_router
from kadastra.config import Settings
from kadastra.ml.train import CatBoostParams
from kadastra.ports.model_registry import ModelRegistryPort
from kadastra.usecases.build_buildings_features import BuildBuildingsFeatures
from kadastra.usecases.build_gold_features import BuildGoldFeatures
from kadastra.usecases.build_metro_features import BuildMetroFeatures
from kadastra.usecases.build_region_coverage import BuildRegionCoverage
from kadastra.usecases.build_road_features import BuildRoadFeatures
from kadastra.usecases.build_synthetic_target import BuildSyntheticTarget
from kadastra.usecases.get_hex_features import GetHexFeatures
from kadastra.usecases.train_valuation_model import TrainValuationModel
from kadastra.web.routes import make_web_router


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
            raise RuntimeError(
                "S3 credentials not configured: set S3_BUCKET, S3_ACCESS_KEY, S3_SECRET_KEY in .env"
            )
        return S3RawData(
            bucket=s.s3_bucket,
            access_key=s.s3_access_key,
            secret_key=s.s3_secret_key,
            endpoint_url=s.s3_endpoint_url,
            region=s.s3_region,
            addressing_style=s.s3_addressing_style,
        )

    def build_metro_features(self) -> BuildMetroFeatures:
        s = self._settings
        return BuildMetroFeatures(
            coverage_reader=ParquetCoverageStore(s.coverage_store_path),
            raw_data=self.build_s3_raw_data(),
            feature_store=ParquetFeatureStore(s.feature_store_path),
            stations_key=s.metro_stations_key,
            entrances_key=s.metro_entrances_key,
        )

    def build_buildings_features(self) -> BuildBuildingsFeatures:
        s = self._settings
        return BuildBuildingsFeatures(
            coverage_reader=ParquetCoverageStore(s.coverage_store_path),
            raw_data=self.build_s3_raw_data(),
            feature_store=ParquetFeatureStore(s.feature_store_path),
            buildings_key=s.buildings_key,
        )

    def build_road_features(self) -> BuildRoadFeatures:
        s = self._settings
        return BuildRoadFeatures(
            coverage_reader=ParquetCoverageStore(s.coverage_store_path),
            raw_data=self.build_s3_raw_data(),
            feature_store=ParquetFeatureStore(s.feature_store_path),
            roads_key=s.roads_key,
        )

    def build_gold_features(self) -> BuildGoldFeatures:
        s = self._settings
        return BuildGoldFeatures(
            coverage_reader=ParquetCoverageStore(s.coverage_store_path),
            feature_reader=ParquetFeatureStore(s.feature_store_path),
            gold_store=ParquetGoldFeatureStore(s.gold_store_path),
            feature_sets=s.gold_feature_sets,
        )

    def build_get_hex_features(self) -> GetHexFeatures:
        return GetHexFeatures(ParquetGoldFeatureStore(self._settings.gold_store_path))

    def build_synthetic_target(self) -> BuildSyntheticTarget:
        s = self._settings
        return BuildSyntheticTarget(
            gold_reader=ParquetGoldFeatureStore(s.gold_store_path),
            target_store=ParquetGoldFeatureStore(s.synthetic_target_store_path),
            seed=s.synthetic_target_seed,
        )

    def build_model_registry(self) -> ModelRegistryPort:
        s = self._settings
        if s.mlflow_enabled:
            if not s.mlflow_tracking_uri:
                raise RuntimeError(
                    "MLFLOW_TRACKING_URI is required when MLFLOW_ENABLED=True"
                )
            return MLflowModelRegistry(
                tracking_uri=s.mlflow_tracking_uri,
                experiment_name=s.mlflow_experiment_name,
            )
        return LocalModelRegistry(s.model_registry_path)

    def build_train_valuation_model(self) -> TrainValuationModel:
        s = self._settings
        params = CatBoostParams(
            iterations=s.catboost_iterations,
            learning_rate=s.catboost_learning_rate,
            depth=s.catboost_depth,
            seed=s.catboost_seed,
        )
        return TrainValuationModel(
            gold_reader=ParquetGoldFeatureStore(s.gold_store_path),
            target_reader=ParquetGoldFeatureStore(s.synthetic_target_store_path),
            model_registry=self.build_model_registry(),
            params=params,
            n_splits=s.train_n_splits,
            parent_resolution=s.train_parent_resolution,
        )


def create_app(settings: Settings) -> FastAPI:
    container = Container(settings)
    templates_dir = Path(__file__).parent / "web" / "templates"

    app = FastAPI(title="kadastra")
    app.include_router(make_api_router(container.build_get_hex_features(), settings.region_code))
    app.include_router(make_web_router(templates_dir))
    return app
