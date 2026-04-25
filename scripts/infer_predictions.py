"""Run inference: load latest baseline model, predict on gold features, save parquet.

If the env var `INFER_RUN_ID` is set, that run is used; otherwise the latest run
matching the baseline run-name prefix at the chosen resolution is selected.
"""

import os

from kadastra.composition_root import Container
from kadastra.config import Settings


def main() -> None:
    settings = Settings()
    container = Container(settings)
    usecase = container.build_infer_valuation()

    resolution = settings.h3_resolutions[-1]
    run_id_override = os.environ.get("INFER_RUN_ID") or None
    backend = "mlflow" if settings.mlflow_enabled else "local"

    print(
        f"Inferring predictions: region={settings.region_code} resolution={resolution} "
        f"backend={backend} run_id={run_id_override or '(latest)'} "
        f"out={settings.predictions_store_path}"
    )

    used = usecase.execute(settings.region_code, resolution, run_id=run_id_override)
    print(f"Done. used run_id={used}")


if __name__ == "__main__":
    main()
