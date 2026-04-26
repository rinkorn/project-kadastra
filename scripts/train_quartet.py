"""Train the Black/Grey/White/Naive quartet (ADR-0016) per AssetClass.

Run name: quartet-object-{class}. Registry backend follows
MLFLOW_ENABLED. Each run carries the four models' OOF predictions,
serialized White/Grey/Naive models, and a quartet_metrics.json with
per-model MAE/RMSE/MAPE/Spearman + percentile asymmetry +
loss_on_simplification (in pp).
"""

from kadastra.composition_root import Container
from kadastra.config import Settings
from kadastra.domain.asset_class import AssetClass


def main() -> None:
    settings = Settings()
    container = Container(settings)
    usecase = container.build_train_quartet()

    backend = "mlflow" if settings.mlflow_enabled else "local"
    classes = list(AssetClass)
    print(
        f"Training quartet (Black/White/Grey/Naive): "
        f"region={settings.region_code} "
        f"classes={[c.value for c in classes]} "
        f"catboost(iters={settings.catboost_iterations}, "
        f"lr={settings.catboost_learning_rate}, depth={settings.catboost_depth}) "
        f"ebm(max_bins={settings.ebm_max_bins}, "
        f"interactions={settings.ebm_interactions}) "
        f"grey(max_depth={settings.grey_tree_max_depth}) "
        f"n_splits={settings.train_n_splits} "
        f"parent_res={settings.train_parent_resolution} backend={backend} "
        f"parallel_folds={settings.quartet_parallel_folds} "
        f"skip_final_fits={settings.quartet_skip_final_simplifier_fits}"
    )

    for asset_class in classes:
        run_id = usecase.execute(settings.region_code, asset_class)
        print(f"  class={asset_class.value}: run_id={run_id}")


if __name__ == "__main__":
    main()
