"""Train one CatBoost model per AssetClass on the per-object table.

Run name: catboost-object-{class}. Registry backend follows
MLFLOW_ENABLED.
"""

from kadastra.composition_root import Container
from kadastra.config import Settings
from kadastra.domain.asset_class import AssetClass


def main() -> None:
    settings = Settings()
    container = Container(settings)
    usecase = container.build_train_object_valuation_model()

    backend = "mlflow" if settings.mlflow_enabled else "local"
    classes = list(AssetClass)
    print(
        f"Training object valuation models: region={settings.region_code} "
        f"classes={[c.value for c in classes]} "
        f"iters={settings.catboost_iterations} lr={settings.catboost_learning_rate} "
        f"depth={settings.catboost_depth} n_splits={settings.train_n_splits} "
        f"parent_res={settings.train_parent_resolution} backend={backend}"
    )

    for asset_class in classes:
        run_id = usecase.execute(settings.region_code, asset_class)
        print(f"  class={asset_class.value}: run_id={run_id}")


if __name__ == "__main__":
    main()
