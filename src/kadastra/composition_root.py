from kadastra.adapters.local_geojson_region_boundary import LocalGeoJsonRegionBoundary
from kadastra.adapters.parquet_coverage_store import ParquetCoverageStore
from kadastra.adapters.s3_raw_data import S3RawData
from kadastra.config import Settings
from kadastra.usecases.build_region_coverage import BuildRegionCoverage


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
