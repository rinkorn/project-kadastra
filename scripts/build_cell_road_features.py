"""Run BuildCellRoadFeatures end-to-end using Settings (Слой 1 ЦОФ)."""

from kadastra.composition_root import Container
from kadastra.config import Settings


def main() -> None:
    settings = Settings()
    container = Container(settings)
    usecase = container.build_cell_road_features()

    print(
        f"Building cell road-density ЦОФ: region={settings.region_code} "
        f"resolution={settings.cell_tsorf_resolution} "
        f"radius_m={settings.object_road_radius_m}"
    )
    usecase.execute(settings.region_code, settings.cell_tsorf_resolution)
    print("done")


if __name__ == "__main__":
    main()
