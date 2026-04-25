"""Run inference for every configured resolution: load latest baseline model,
predict on gold features, save parquet snapshot.

If the env var `INFER_RUN_ID` is set, it is used for all resolutions (caveat:
a model trained at one resolution must not be applied to features of another;
override is intended for the single-resolution case).
"""

import os

from kadastra.composition_root import Container
from kadastra.config import Settings


def main() -> None:
    settings = Settings()
    container = Container(settings)
    usecase = container.build_infer_valuation()

    run_id_override = os.environ.get("INFER_RUN_ID") or None
    backend = "mlflow" if settings.mlflow_enabled else "local"
    print(
        f"Inferring predictions: region={settings.region_code} "
        f"resolutions={settings.h3_resolutions} backend={backend} "
        f"run_id={run_id_override or '(latest per resolution)'} "
        f"out={settings.predictions_store_path}"
    )

    for resolution in settings.h3_resolutions:
        used = usecase.execute(settings.region_code, resolution, run_id=run_id_override)
        print(f"  res={resolution}: used run_id={used}")


if __name__ == "__main__":
    main()
