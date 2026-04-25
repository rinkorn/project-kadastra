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
