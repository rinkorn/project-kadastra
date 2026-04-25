"""Enrich per-object tables with metro / road / neighbor features.

Reads each asset_class partition, computes context features against
all classes combined, and writes the enriched partition back.
"""

from kadastra.composition_root import Container
from kadastra.config import Settings
from kadastra.domain.asset_class import AssetClass


def main() -> None:
    settings = Settings()
    container = Container(settings)
    usecase = container.build_object_features()

    classes = list(AssetClass)
    print(
        f"Computing object features: region={settings.region_code} "
        f"classes={[c.value for c in classes]} "
        f"neighbor_radius_m={settings.object_neighbor_radius_m} "
        f"road_radius_m={settings.object_road_radius_m}"
    )
    usecase.execute(settings.region_code, asset_classes=classes)
    print("done")


if __name__ == "__main__":
    main()
