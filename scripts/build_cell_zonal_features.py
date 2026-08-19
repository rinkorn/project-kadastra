"""Run BuildCellZonalFeatures end-to-end using Settings (Слой 1 ЦОФ, self-free)."""

from kadastra.composition_root import Container
from kadastra.config import Settings


def main() -> None:
    settings = Settings()
    container = Container(settings)
    usecase = container.build_cell_zonal_features()

    print(
        f"Building cell zonal-density ЦОФ: region={settings.region_code} "
        f"resolution={settings.cell_tsorf_resolution} "
        f"radii_m={settings.zonal_radii_m}"
    )
    usecase.execute(settings.region_code, settings.cell_tsorf_resolution)
    print("done")


if __name__ == "__main__":
    main()
