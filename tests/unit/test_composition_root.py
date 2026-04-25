from pathlib import Path

import pytest

from kadastra.adapters.s3_raw_data import S3RawData
from kadastra.composition_root import Container
from kadastra.config import Settings
from kadastra.usecases.build_region_coverage import BuildRegionCoverage


def test_container_builds_region_coverage_usecase(tmp_path: Path) -> None:
    settings = Settings(
        region_boundary_path=tmp_path / "boundary.geojson",
        coverage_store_path=tmp_path / "coverage",
    )
    container = Container(settings)

    usecase = container.build_region_coverage()

    assert isinstance(usecase, BuildRegionCoverage)


def test_settings_defaults_match_pilot_region() -> None:
    settings = Settings()

    assert settings.region_code == "RU-TA"
    assert settings.h3_resolutions == [7, 8]
    assert settings.region_boundary_field == "shapeISO"


def test_container_builds_s3_raw_data_with_credentials(tmp_path: Path) -> None:
    settings = Settings(
        region_boundary_path=tmp_path / "b.geojson",
        coverage_store_path=tmp_path / "c",
        s3_endpoint_url="https://example.com",
        s3_bucket="bucket",
        s3_access_key="a",
        s3_secret_key="s",
    )
    container = Container(settings)

    adapter = container.build_s3_raw_data()

    assert isinstance(adapter, S3RawData)


def test_container_raises_when_s3_credentials_missing(tmp_path: Path) -> None:
    settings = Settings(
        region_boundary_path=tmp_path / "b.geojson",
        coverage_store_path=tmp_path / "c",
        s3_bucket=None,
        s3_access_key=None,
        s3_secret_key=None,
    )
    container = Container(settings)

    with pytest.raises(RuntimeError, match="S3 credentials"):
        container.build_s3_raw_data()
