from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    region_code: str = "RU-TA"
    region_boundary_path: Path = Path("data/raw/regions/geoBoundaries-RUS-ADM1.geojson")
    region_boundary_field: str = "shapeISO"
    coverage_store_path: Path = Path("data/silver/coverage")
    h3_resolutions: list[int] = [7, 8]

    s3_endpoint_url: str | None = None
    s3_bucket: str | None = None
    s3_access_key: str | None = None
    s3_secret_key: str | None = None
    s3_region: str = "us-east-1"
    s3_addressing_style: str = "path"

    feature_store_path: Path = Path("data/silver/features")
    metro_stations_key: str = "Kadatastr/metro/metro_stations.csv"
    metro_entrances_key: str = "Kadatastr/metro/metro_entrances.csv"
    buildings_key: str = "Kadatastr/rosreestr/osm_buildings_kazan.csv"
    roads_key: str = "Kadatastr/tatarstan_major_roads/tatarstan_major_roads.json"

    gold_store_path: Path = Path("data/gold/features")
    gold_feature_sets: list[str] = ["metro", "buildings", "roads"]

    synthetic_target_store_path: Path = Path("data/gold/targets")
    synthetic_target_seed: int = 42

    model_registry_path: Path = Path("data/models")
    catboost_iterations: int = 500
    catboost_learning_rate: float = 0.05
    catboost_depth: int = 6
    catboost_seed: int = 42
    train_n_splits: int = 5
    train_parent_resolution: int = 6

    mlflow_enabled: bool = False
    mlflow_tracking_uri: str | None = None
    mlflow_experiment_name: str = "kadastra-valuation"
