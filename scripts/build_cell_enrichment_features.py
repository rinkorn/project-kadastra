"""Build the Слой 1 ``enrichment`` feature set per cell (ADR-0029).

Computes the ADR-0021..0025 location features (DEM, road-class, CBD,
isochrone join, heritage, ЗОУИТ, territory/OKTMO + macro) for every
anchor cell of the region coverage and stores them as
``feature_set=enrichment`` in the cell feature store.

Запуск:
    uv run python scripts/build_cell_enrichment_features.py
"""

from __future__ import annotations

from kadastra.composition_root import Container
from kadastra.config import Settings


def main() -> int:
    settings = Settings()
    container = Container(settings)
    usecase = container.build_cell_enrichment_features()

    print(f"Building cell enrichment ЦОФ: region={settings.region_code} resolution={settings.cell_tsorf_resolution}")
    usecase.execute(settings.region_code, settings.cell_tsorf_resolution)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
