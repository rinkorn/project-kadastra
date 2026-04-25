"""Run BuildRoadFeatures end-to-end: read coverage and roads JSON, write feature parquet."""

from kadastra.composition_root import Container
from kadastra.config import Settings


def main() -> None:
    settings = Settings()
    container = Container(settings)
    usecase = container.build_road_features()

    print(
        f"Building road features for region={settings.region_code} "
        f"resolutions={settings.h3_resolutions} "
        f"feature_store={settings.feature_store_path}"
    )
    usecase.execute(settings.region_code, settings.h3_resolutions)
    print(f"Road features saved under {settings.feature_store_path}")


if __name__ == "__main__":
    main()
