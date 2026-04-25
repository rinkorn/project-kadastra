"""Append per-class synthetic target column to each object partition.

Reads featured objects, runs the asset-class-aware formula
(compute_object_synthetic_target), writes the partition back with
synthetic_target_rub_per_m2.
"""

from kadastra.composition_root import Container
from kadastra.config import Settings
from kadastra.domain.asset_class import AssetClass


def main() -> None:
    settings = Settings()
    container = Container(settings)
    usecase = container.build_object_synthetic_target()

    classes = list(AssetClass)
    print(
        f"Computing object synthetic target: region={settings.region_code} "
        f"classes={[c.value for c in classes]} "
        f"seed={settings.synthetic_target_seed}"
    )
    usecase.execute(settings.region_code, asset_classes=classes)
    print("done")


if __name__ == "__main__":
    main()
