from dataclasses import asdict

import numpy as np
import polars as pl

from kadastra.ml.train import CatBoostParams, cross_validate, train_catboost
from kadastra.ports.gold_feature_reader import GoldFeatureReaderPort
from kadastra.ports.model_registry import ModelRegistryPort

_KEY_COLUMNS = ["h3_index", "resolution"]
_TARGET_COLUMN = "synthetic_target_rub_per_m2"


class TrainValuationModel:
    def __init__(
        self,
        gold_reader: GoldFeatureReaderPort,
        target_reader: GoldFeatureReaderPort,
        model_registry: ModelRegistryPort,
        params: CatBoostParams,
        n_splits: int,
        parent_resolution: int,
    ) -> None:
        self._gold_reader = gold_reader
        self._target_reader = target_reader
        self._model_registry = model_registry
        self._params = params
        self._n_splits = n_splits
        self._parent_resolution = parent_resolution

    def execute(self, region_code: str, resolution: int) -> str:
        gold = self._gold_reader.load(region_code, resolution)
        target = self._target_reader.load(region_code, resolution).select(
            [*_KEY_COLUMNS, _TARGET_COLUMN]
        )

        df = gold.join(target, on=_KEY_COLUMNS, how="inner").drop_nulls(subset=[_TARGET_COLUMN])

        feature_cols = [c for c in df.columns if c not in {*_KEY_COLUMNS, _TARGET_COLUMN}]
        df = df.with_columns([pl.col(c).fill_null(0).cast(pl.Float64) for c in feature_cols])

        X = df.select(feature_cols).to_numpy().astype(np.float64)
        y = df[_TARGET_COLUMN].to_numpy().astype(np.float64)
        h3_indices = df["h3_index"].to_list()

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
            "n_splits": self._n_splits,
            "parent_resolution": self._parent_resolution,
            "feature_columns": feature_cols,
            "n_samples": len(y),
        }
        metrics_payload = {k: v for k, v in cv.items() if isinstance(v, float)}

        return self._model_registry.log_run(
            run_name=f"catboost-baseline-res{resolution}",
            params=params_payload,
            metrics=metrics_payload,
            model=final_model,
        )
