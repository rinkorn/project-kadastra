"""Train a CatBoost baseline on gold features + synthetic target with spatial K-fold CV.

Iterates over all `h3_resolutions` configured in Settings.
Registry: MLflow if MLFLOW_ENABLED=True, else local file registry.
"""

from kadastra.composition_root import Container
from kadastra.config import Settings


def main() -> None:
    settings = Settings()
    container = Container(settings)
    usecase = container.build_train_valuation_model()

    backend = "mlflow" if settings.mlflow_enabled else "local"
    print(
        f"Training valuation model: region={settings.region_code} "
        f"resolutions={settings.h3_resolutions} "
        f"iters={settings.catboost_iterations} lr={settings.catboost_learning_rate} "
        f"depth={settings.catboost_depth} n_splits={settings.train_n_splits} "
        f"parent_res={settings.train_parent_resolution} backend={backend}"
    )

    for resolution in settings.h3_resolutions:
        run_id = usecase.execute(settings.region_code, resolution)
        print(f"  res={resolution}: run_id={run_id}")


if __name__ == "__main__":
    main()
