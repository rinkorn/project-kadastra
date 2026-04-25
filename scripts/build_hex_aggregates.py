"""Aggregate per-object gold + OOF predictions to per-hex parquet.

For each requested resolution, writes one parquet at
``data/gold/hex_aggregates/region={REGION}/resolution={R}/data.parquet``
with one row per (h3_index, asset_class) — including a cross-class
``"all"`` roll-up. The map UI's hex-mode reads from these files.

Run **after** training (which populates OOF predictions per class). If
training has not run yet, the prediction columns will be nulls and the
hex view shows medians of the ЕГРН target only.
"""

from kadastra.composition_root import Container
from kadastra.config import Settings
from kadastra.domain.asset_class import AssetClass


def main() -> None:
    settings = Settings()
    container = Container(settings)
    usecase = container.build_hex_aggregates()

    classes = list(AssetClass)
    print(
        f"Aggregating objects to hex: region={settings.region_code} "
        f"classes={[c.value for c in classes]} "
        f"resolutions={settings.hex_aggregates_resolutions}"
    )
    usecase.execute(settings.region_code, asset_classes=classes)
    print("done")


if __name__ == "__main__":
    main()
