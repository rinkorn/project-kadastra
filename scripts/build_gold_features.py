"""Run BuildGoldFeatures end-to-end: join all silver feature sets into a single gold parquet."""

from kadastra.composition_root import Container
from kadastra.config import Settings


def main() -> None:
    settings = Settings()
    container = Container(settings)
    usecase = container.build_gold_features()

    print(
        f"Building gold features for region={settings.region_code} "
        f"resolutions={settings.h3_resolutions} "
        f"feature_sets={settings.gold_feature_sets} "
        f"gold_store={settings.gold_store_path}"
    )
    usecase.execute(settings.region_code, settings.h3_resolutions)
    print(f"Gold features saved under {settings.gold_store_path}")


if __name__ == "__main__":
    main()
