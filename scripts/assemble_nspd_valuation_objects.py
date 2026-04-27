"""Assemble silver NSPD frames into per-class valuation_object partitions.

Reads ``data/silver/nspd/region={code}/source={buildings|landplots}/data.parquet``,
maps to the existing valuation_object schema (legacy column name
``synthetic_target_rub_per_m2`` is kept per ADR-0009 — it now holds the
real ЕГРН ``cost_index`` rather than a synthetic proxy), drops rows
without an asset_class or target, and writes one partition per class
to ``data/gold/valuation_objects/region={code}/asset_class={class}/``.
"""

from kadastra.composition_root import Container
from kadastra.config import Settings
from kadastra.domain.asset_class import AssetClass


def main() -> None:
    settings = Settings()
    container = Container(settings)
    usecase = container.build_assemble_nspd_valuation_objects()

    classes = list(AssetClass)
    print(f"Assembling NSPD valuation objects: region={settings.region_code} classes={[c.value for c in classes]}")
    usecase.execute(settings.region_code, asset_classes=classes)
    print("done")


if __name__ == "__main__":
    main()
