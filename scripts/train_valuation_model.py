"""Train a CatBoost baseline on gold features + synthetic target with spatial K-fold CV.

Resolution defaults to the primary one (last entry of `h3_resolutions`).
Registry: MLflow if MLFLOW_ENABLED=True, else local file registry.
"""

from kadastra.composition_root import Container
from kadastra.config import Settings


def main() -> None:
    settings = Settings()
    container = Container(settings)
    usecase = container.build_train_valuation_model()

    resolution = settings.h3_resolutions[-1]
    backend = "mlflow" if settings.mlflow_enabled else "local"

    print(
        f"Training valuation model: region={settings.region_code} resolution={resolution} "
        f"iters={settings.catboost_iterations} lr={settings.catboost_learning_rate} "
        f"depth={settings.catboost_depth} n_splits={settings.train_n_splits} "
        f"parent_res={settings.train_parent_resolution} backend={backend}"
    )

    run_id = usecase.execute(settings.region_code, resolution)
    print(f"Done. run_id={run_id}")


if __name__ == "__main__":
    main()
