"""Run BuildCellMetroFeatures end-to-end using Settings (Слой 1 ЦОФ, graph)."""

from kadastra.composition_root import Container
from kadastra.config import Settings


def main() -> None:
    settings = Settings()
    container = Container(settings)
    usecase = container.build_cell_metro_features()

    print(f"Building cell metro ЦОФ (graph): region={settings.region_code} resolution={settings.cell_tsorf_resolution}")
    usecase.execute(settings.region_code, settings.cell_tsorf_resolution)
    print("done")


if __name__ == "__main__":
    main()
