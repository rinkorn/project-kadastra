"""Run BuildCellPolygonFeatures end-to-end using Settings (Слой 1 ЦОФ)."""

from kadastra.composition_root import Container
from kadastra.config import Settings


def main() -> None:
    settings = Settings()
    container = Container(settings)
    usecase = container.build_cell_polygon_features()

    print(f"Building cell polygon-share ЦОФ: region={settings.region_code} resolution={settings.cell_tsorf_resolution}")
    usecase.execute(settings.region_code, settings.cell_tsorf_resolution)
    print("done")


if __name__ == "__main__":
    main()
