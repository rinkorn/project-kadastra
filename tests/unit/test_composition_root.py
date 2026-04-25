from pathlib import Path

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
