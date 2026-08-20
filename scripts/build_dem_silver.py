"""Preprocess raw DEM tiles into silver topographic layers (ADR-0023).

Merges the Copernicus GLO-30 tiles in ``dem_raw_dir``, reprojects the
mosaic to the region's UTM zone and writes elevation / slope /
relative-relief rasters under
``data/silver/dem/region={code}/``. Consumed by BuildObjectFeatures
via RasterioDemSampler.
"""

from kadastra.composition_root import Container
from kadastra.config import Settings


def main() -> None:
    settings = Settings()
    container = Container(settings)
    usecase = container.build_dem_silver()

    print(
        f"Building DEM silver layers: region={settings.region_code} "
        f"raw_dir={settings.dem_raw_dir} "
        f"relief_radius_m={settings.dem_relief_radius_m}"
    )
    usecase.execute(settings.region_code)
    print("done")


if __name__ == "__main__":
    main()
