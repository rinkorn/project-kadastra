"""Run per-class inference: load latest catboost-object-{class} model,
predict on featured objects, save predictions parquet.

Output: data/gold/object_predictions/region={code}/asset_class={class}/data.parquet
columns: object_id, asset_class, lat, lon, predicted_value
"""

from kadastra.composition_root import Container
from kadastra.config import Settings
from kadastra.domain.asset_class import AssetClass


def main() -> None:
    settings = Settings()
    container = Container(settings)
    usecase = container.build_infer_object_valuation()

    backend = "mlflow" if settings.mlflow_enabled else "local"
    classes = list(AssetClass)
    print(
        f"Inferring object predictions: region={settings.region_code} "
        f"classes={[c.value for c in classes]} backend={backend} "
        f"out={settings.object_predictions_store_path}"
    )

    for asset_class in classes:
        used = usecase.execute(settings.region_code, asset_class)
        print(f"  class={asset_class.value}: used run_id={used}")


if __name__ == "__main__":
    main()
