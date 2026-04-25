from dataclasses import asdict

import h3
import numpy as np
import polars as pl

from kadastra.domain.asset_class import AssetClass
from kadastra.ml.object_feature_matrix import build_object_feature_matrix
from kadastra.ml.train import CatBoostParams, cross_validate, train_catboost
from kadastra.ports.model_registry import ModelRegistryPort
from kadastra.ports.valuation_object_reader import ValuationObjectReaderPort

_TARGET_COLUMN = "synthetic_target_rub_per_m2"
_NON_FEATURE_COLUMNS = frozenset(
    {
        "object_id",
        "asset_class",
        "lat",
        "lon",
        _TARGET_COLUMN,
        "cost_value_rub",
    }
)
def _is_numeric(dtype: pl.DataType) -> bool:
    return dtype.is_numeric()


def _is_categorical(dtype: pl.DataType) -> bool:
    return dtype == pl.Utf8 or dtype == pl.Categorical


class TrainObjectValuationModel:
    def __init__(
        self,
        reader: ValuationObjectReaderPort,
        model_registry: ModelRegistryPort,
        params: CatBoostParams,
        n_splits: int,
        parent_resolution: int,
    ) -> None:
        self._reader = reader
        self._model_registry = model_registry
        self._params = params
        self._n_splits = n_splits
        self._parent_resolution = parent_resolution

    def execute(self, region_code: str, asset_class: AssetClass) -> str:
        df = self._reader.load(region_code, asset_class).drop_nulls(
            subset=[_TARGET_COLUMN]
        )

        numeric_cols = [
            c
            for c in df.columns
            if c not in _NON_FEATURE_COLUMNS and _is_numeric(df.schema[c])
        ]
        categorical_cols = [
            c
            for c in df.columns
            if c not in _NON_FEATURE_COLUMNS and _is_categorical(df.schema[c])
        ]
        feature_cols = numeric_cols + categorical_cols
        cat_feature_indices = list(range(len(numeric_cols), len(feature_cols)))

        X = build_object_feature_matrix(
            df, numeric_cols=numeric_cols, categorical_cols=categorical_cols
        )
        y = df[_TARGET_COLUMN].to_numpy().astype(np.float64)

        cell_resolution = max(self._parent_resolution + 1, 10)
        h3_indices = [
            h3.latlng_to_cell(float(lat), float(lon), cell_resolution)
            for lat, lon in zip(df["lat"].to_list(), df["lon"].to_list(), strict=True)
        ]

        cv = cross_validate(
            X,
            y,
            h3_indices,
            params=self._params,
            n_splits=self._n_splits,
            parent_resolution=self._parent_resolution,
            cat_features=cat_feature_indices or None,
        )
        final_model = train_catboost(
            X, y, self._params, cat_features=cat_feature_indices or None
        )

        params_payload = {
            **asdict(self._params),
            "asset_class": asset_class.value,
            "n_splits": self._n_splits,
            "parent_resolution": self._parent_resolution,
            "cell_resolution": cell_resolution,
            "feature_columns": feature_cols,
            "cat_feature_indices": cat_feature_indices,
            "n_samples": len(y),
        }
        metrics_payload = {k: v for k, v in cv.items() if isinstance(v, float)}

        return self._model_registry.log_run(
            run_name=f"catboost-object-{asset_class.value}",
            params=params_payload,
            metrics=metrics_payload,
            model=final_model,
        )
