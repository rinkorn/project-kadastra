"""Run BuildRegionCoverage end-to-end using Settings."""

from kadastra.composition_root import Container
from kadastra.config import Settings


def main() -> None:
    settings = Settings()
    container = Container(settings)
    usecase = container.build_region_coverage()

    print(
        f"Building coverage for region={settings.region_code} "
        f"resolutions={settings.h3_resolutions} "
        f"boundary={settings.region_boundary_path} "
        f"store={settings.coverage_store_path}"
    )
    usecase.execute(settings.region_code, settings.h3_resolutions)
    print(f"Coverage saved under {settings.coverage_store_path}")


if __name__ == "__main__":
    main()
