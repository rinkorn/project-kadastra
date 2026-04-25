"""Materialize per-object valuation tables from the OSM buildings CSV.

Output: data/gold/valuation_objects/region={code}/asset_class={class}/data.parquet
"""

from kadastra.composition_root import Container
from kadastra.config import Settings
from kadastra.domain.asset_class import AssetClass


def main() -> None:
    settings = Settings()
    container = Container(settings)
    usecase = container.build_valuation_objects()

    classes = list(AssetClass)
    print(
        f"Building valuation objects: region={settings.region_code} "
        f"classes={[c.value for c in classes]} "
        f"key={settings.buildings_key}"
    )
    usecase.execute(settings.region_code, asset_classes=classes)
    print("done")


if __name__ == "__main__":
    main()
