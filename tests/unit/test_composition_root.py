from pathlib import Path

import pytest

from kadastra.adapters.local_model_loader import LocalModelLoader
from kadastra.adapters.local_model_registry import LocalModelRegistry
from kadastra.adapters.mlflow_model_loader import MLflowModelLoader
from kadastra.adapters.mlflow_model_registry import MLflowModelRegistry
from kadastra.adapters.s3_raw_data import S3RawData
from kadastra.composition_root import Container
from kadastra.config import Settings
from kadastra.usecases.build_region_coverage import BuildRegionCoverage
from kadastra.usecases.build_synthetic_target import BuildSyntheticTarget
from kadastra.usecases.infer_valuation import InferValuation
from kadastra.usecases.train_valuation_model import TrainValuationModel


def test_container_builds_region_coverage_usecase(tmp_path: Path) -> None:
    settings = Settings(
        region_boundary_path=tmp_path / "boundary.geojson",
        coverage_store_path=tmp_path / "coverage",
    )
    container = Container(settings)

    usecase = container.build_region_coverage()

    assert isinstance(usecase, BuildRegionCoverage)


def test_settings_defaults_match_pilot_region() -> None:
    settings = Settings()

    assert settings.region_code == "RU-KAZAN-AGG"
    assert settings.h3_resolutions == [7, 8, 9, 10, 11]
    assert settings.region_boundary_field == "shapeISO"


def test_container_builds_s3_raw_data_with_credentials(tmp_path: Path) -> None:
    settings = Settings(
        region_boundary_path=tmp_path / "b.geojson",
        coverage_store_path=tmp_path / "c",
        s3_endpoint_url="https://example.com",
        s3_bucket="bucket",
        s3_access_key="a",
        s3_secret_key="s",
    )
    container = Container(settings)

    adapter = container.build_s3_raw_data()

    assert isinstance(adapter, S3RawData)


def test_container_raises_when_s3_credentials_missing(tmp_path: Path) -> None:
    settings = Settings(
        region_boundary_path=tmp_path / "b.geojson",
        coverage_store_path=tmp_path / "c",
        s3_bucket=None,
        s3_access_key=None,
        s3_secret_key=None,
    )
    container = Container(settings)

    with pytest.raises(RuntimeError, match="S3 credentials"):
        container.build_s3_raw_data()


def test_container_builds_synthetic_target_usecase(tmp_path: Path) -> None:
    settings = Settings(
        region_boundary_path=tmp_path / "b.geojson",
        coverage_store_path=tmp_path / "c",
        gold_store_path=tmp_path / "gold",
        synthetic_target_store_path=tmp_path / "targets",
    )
    container = Container(settings)

    usecase = container.build_synthetic_target()

    assert isinstance(usecase, BuildSyntheticTarget)


def test_settings_has_synthetic_target_defaults() -> None:
    settings = Settings()

    assert settings.synthetic_target_store_path.as_posix().endswith("data/gold/targets")
    assert settings.synthetic_target_seed == 42


def test_settings_has_training_defaults() -> None:
    settings = Settings()

    assert settings.model_registry_path.as_posix().endswith("data/models")
    assert settings.catboost_iterations > 0
    assert 0 < settings.catboost_learning_rate < 1
    assert settings.catboost_depth > 0
    assert settings.train_n_splits >= 2
    assert settings.train_parent_resolution >= 0


def test_container_builds_train_valuation_model_with_local_registry_when_mlflow_disabled(
    tmp_path: Path,
) -> None:
    settings = Settings(
        region_boundary_path=tmp_path / "b.geojson",
        coverage_store_path=tmp_path / "c",
        gold_store_path=tmp_path / "gold",
        synthetic_target_store_path=tmp_path / "targets",
        model_registry_path=tmp_path / "models",
        mlflow_enabled=False,
    )
    container = Container(settings)

    usecase = container.build_train_valuation_model()

    assert isinstance(usecase, TrainValuationModel)
    # Concretely, the adapter inside should be LocalModelRegistry
    assert isinstance(container.build_model_registry(), LocalModelRegistry)


def test_container_builds_mlflow_registry_when_enabled(tmp_path: Path) -> None:
    settings = Settings(
        region_boundary_path=tmp_path / "b.geojson",
        coverage_store_path=tmp_path / "c",
        mlflow_enabled=True,
        mlflow_tracking_uri=f"file:{tmp_path / 'mlruns'}",
    )
    container = Container(settings)

    registry = container.build_model_registry()

    assert isinstance(registry, MLflowModelRegistry)


def test_container_raises_when_mlflow_enabled_but_uri_missing(tmp_path: Path) -> None:
    settings = Settings(
        region_boundary_path=tmp_path / "b.geojson",
        coverage_store_path=tmp_path / "c",
        mlflow_enabled=True,
        mlflow_tracking_uri=None,
    )
    container = Container(settings)

    with pytest.raises(RuntimeError, match="MLFLOW_TRACKING_URI"):
        container.build_model_registry()


def test_settings_has_predictions_store_default() -> None:
    settings = Settings()

    assert settings.predictions_store_path.as_posix().endswith("data/gold/predictions")


def test_container_builds_local_model_loader_when_mlflow_disabled(tmp_path: Path) -> None:
    settings = Settings(
        region_boundary_path=tmp_path / "b.geojson",
        coverage_store_path=tmp_path / "c",
        model_registry_path=tmp_path / "models",
        mlflow_enabled=False,
    )
    container = Container(settings)

    assert isinstance(container.build_model_loader(), LocalModelLoader)


def test_container_builds_mlflow_model_loader_when_enabled(tmp_path: Path) -> None:
    settings = Settings(
        region_boundary_path=tmp_path / "b.geojson",
        coverage_store_path=tmp_path / "c",
        mlflow_enabled=True,
        mlflow_tracking_uri=f"file:{tmp_path / 'mlruns'}",
    )
    container = Container(settings)

    assert isinstance(container.build_model_loader(), MLflowModelLoader)


def test_container_builds_infer_valuation(tmp_path: Path) -> None:
    settings = Settings(
        region_boundary_path=tmp_path / "b.geojson",
        coverage_store_path=tmp_path / "c",
        gold_store_path=tmp_path / "gold",
        predictions_store_path=tmp_path / "preds",
        model_registry_path=tmp_path / "models",
        mlflow_enabled=False,
    )
    container = Container(settings)

    assert isinstance(container.build_infer_valuation(), InferValuation)


def test_settings_has_object_pipeline_defaults() -> None:
    settings = Settings()

    assert settings.valuation_object_store_path.as_posix().endswith("data/gold/valuation_objects")
    assert settings.object_predictions_store_path.as_posix().endswith("data/gold/object_predictions")
    assert settings.object_neighbor_radius_m == 500.0
    assert settings.object_road_radius_m == 500.0


def test_container_builds_build_valuation_objects(tmp_path: Path) -> None:
    from kadastra.usecases.build_valuation_objects import BuildValuationObjects

    settings = Settings(
        region_boundary_path=tmp_path / "b.geojson",
        coverage_store_path=tmp_path / "c",
        valuation_object_store_path=tmp_path / "objects",
        s3_endpoint_url="https://example.com",
        s3_bucket="b",
        s3_access_key="k",
        s3_secret_key="s",
    )
    container = Container(settings)

    assert isinstance(container.build_valuation_objects(), BuildValuationObjects)


def test_container_builds_build_object_features(tmp_path: Path) -> None:
    import polars as pl

    from kadastra.usecases.build_object_features import BuildObjectFeatures

    edges_path = tmp_path / "edges.parquet"
    edges_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "from_lat": [55.78],
            "from_lon": [49.12],
            "to_lat": [55.79],
            "to_lon": [49.13],
            "length_m": [100.0],
        }
    ).write_parquet(edges_path)

    settings = Settings(
        region_boundary_path=tmp_path / "b.geojson",
        coverage_store_path=tmp_path / "c",
        valuation_object_store_path=tmp_path / "objects",
        road_graph_edges_path=edges_path,
        s3_endpoint_url="https://example.com",
        s3_bucket="b",
        s3_access_key="k",
        s3_secret_key="s",
    )
    container = Container(settings)

    assert isinstance(container.build_object_features(), BuildObjectFeatures)


def test_container_builds_build_object_synthetic_target(tmp_path: Path) -> None:
    from kadastra.usecases.build_object_synthetic_target import BuildObjectSyntheticTarget

    settings = Settings(
        region_boundary_path=tmp_path / "b.geojson",
        coverage_store_path=tmp_path / "c",
        valuation_object_store_path=tmp_path / "objects",
    )
    container = Container(settings)

    assert isinstance(container.build_object_synthetic_target(), BuildObjectSyntheticTarget)


def test_container_builds_train_object_valuation_model(tmp_path: Path) -> None:
    from kadastra.usecases.train_object_valuation_model import TrainObjectValuationModel

    settings = Settings(
        region_boundary_path=tmp_path / "b.geojson",
        coverage_store_path=tmp_path / "c",
        valuation_object_store_path=tmp_path / "objects",
        model_registry_path=tmp_path / "models",
        mlflow_enabled=False,
    )
    container = Container(settings)

    assert isinstance(container.build_train_object_valuation_model(), TrainObjectValuationModel)


def test_container_builds_infer_object_valuation(tmp_path: Path) -> None:
    from kadastra.usecases.infer_object_valuation import InferObjectValuation

    settings = Settings(
        region_boundary_path=tmp_path / "b.geojson",
        coverage_store_path=tmp_path / "c",
        valuation_object_store_path=tmp_path / "objects",
        object_predictions_store_path=tmp_path / "object_preds",
        model_registry_path=tmp_path / "models",
        mlflow_enabled=False,
    )
    container = Container(settings)

    assert isinstance(container.build_infer_object_valuation(), InferObjectValuation)
