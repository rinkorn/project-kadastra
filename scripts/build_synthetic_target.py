"""Build synthetic price-per-m2 target from the existing gold features table.

Reads gold features, applies compute_synthetic_target (exponential decay from
Kazan + metro/buildings boosts + gaussian noise), writes to the target store.
"""

from kadastra.composition_root import Container
from kadastra.config import Settings


def main() -> None:
    settings = Settings()
    container = Container(settings)
    usecase = container.build_synthetic_target()

    print(
        f"Building synthetic target for region={settings.region_code} "
        f"resolutions={settings.h3_resolutions} "
        f"seed={settings.synthetic_target_seed} "
        f"target_store={settings.synthetic_target_store_path}"
    )
    usecase.execute(settings.region_code, settings.h3_resolutions)
    print(f"Synthetic target saved under {settings.synthetic_target_store_path}")


if __name__ == "__main__":
    main()
