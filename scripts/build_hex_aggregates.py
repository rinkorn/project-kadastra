"""Aggregate per-object gold + OOF predictions to per-hex parquet.

For each (resolution, model) pair writes one parquet at
``data/gold/hex_aggregates/region={REGION}/resolution={R}/model={MODEL}/data.parquet``
with one row per (h3_index, asset_class) — including a cross-class
``"all"`` roll-up. The map UI's hex-mode reads from these files,
selecting partition by the active model dropdown.

Iterates over all 4 ADR-0016 quartet models. If a particular
(class, model) OOF artifact is missing, the corresponding partition
still gets written but ``median_pred_oof_rub_per_m2`` will be null
in those rows.
"""

from kadastra.composition_root import Container
from kadastra.config import Settings
from kadastra.domain.asset_class import AssetClass

_QUARTET_MODELS = ("catboost", "ebm", "grey_tree", "naive_linear")


def main() -> None:
    settings = Settings()
    container = Container(settings)
    usecase = container.build_hex_aggregates()

    classes = list(AssetClass)
    print(
        f"Aggregating objects to hex: region={settings.region_code} "
        f"classes={[c.value for c in classes]} "
        f"resolutions={settings.hex_aggregates_resolutions} "
        f"models={list(_QUARTET_MODELS)}"
    )
    for model in _QUARTET_MODELS:
        print(f"  → model={model}")
        usecase.execute(settings.region_code, classes, model=model)
    print("done")


if __name__ == "__main__":
    main()
