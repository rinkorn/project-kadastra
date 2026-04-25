from dataclasses import asdict

import h3
import numpy as np
import polars as pl

from kadastra.domain.asset_class import AssetClass
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
    }
)


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

        feature_cols = [c for c in df.columns if c not in _NON_FEATURE_COLUMNS]
        df = df.with_columns(
            [pl.col(c).fill_null(0).cast(pl.Float64) for c in feature_cols]
        )

        X = df.select(feature_cols).to_numpy().astype(np.float64)
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
        )
        final_model = train_catboost(X, y, self._params)

        params_payload = {
            **asdict(self._params),
            "asset_class": asset_class.value,
            "n_splits": self._n_splits,
            "parent_resolution": self._parent_resolution,
            "cell_resolution": cell_resolution,
            "feature_columns": feature_cols,
            "n_samples": len(y),
        }
        metrics_payload = {k: v for k, v in cv.items() if isinstance(v, float)}

        return self._model_registry.log_run(
            run_name=f"catboost-object-{asset_class.value}",
            params=params_payload,
            metrics=metrics_payload,
            model=final_model,
        )
