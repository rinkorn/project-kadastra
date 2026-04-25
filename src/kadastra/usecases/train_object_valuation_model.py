import io
from dataclasses import asdict

import h3
import numpy as np
import polars as pl

from kadastra.domain.asset_class import AssetClass
from kadastra.ml.object_feature_columns import select_object_feature_columns
from kadastra.ml.object_feature_matrix import build_object_feature_matrix
from kadastra.ml.train import CatBoostParams, cross_validate, train_catboost
from kadastra.ports.model_registry import ModelRegistryPort
from kadastra.ports.valuation_object_reader import ValuationObjectReaderPort

_TARGET_COLUMN = "synthetic_target_rub_per_m2"
_OOF_ARTIFACT_NAME = "oof_predictions.parquet"


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

        numeric_cols, categorical_cols = select_object_feature_columns(df)
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

        # OOF predictions: each row predicted by a model that did not
        # see it in training (spatial-CV honest view). Joined back to
        # ``object_id`` / ``lat`` / ``lon`` so the inspector can render
        # them on the map without recomputing.
        oof_artifact = _build_oof_artifact(
            df,
            oof_indices=cv["oof_indices"],  # type: ignore[arg-type]
            oof_fold_ids=cv["oof_fold_ids"],  # type: ignore[arg-type]
            oof_y_pred=cv["oof_y_pred"],  # type: ignore[arg-type]
        )

        return self._model_registry.log_run(
            run_name=f"catboost-object-{asset_class.value}",
            params=params_payload,
            metrics=metrics_payload,
            model=final_model,
            artifacts={_OOF_ARTIFACT_NAME: oof_artifact},
        )


def _build_oof_artifact(
    df: pl.DataFrame,
    *,
    oof_indices: list[int],
    oof_fold_ids: list[int],
    oof_y_pred: list[float],
) -> bytes:
    """Serialize OOF predictions as parquet bytes.

    Schema: ``(object_id, lat, lon, fold_id, y_true, y_pred_oof)``.
    Rows are ordered by ``object_id`` for stable lookups downstream.
    """
    selected = df.select(["object_id", "lat", "lon", _TARGET_COLUMN])
    oof_df = pl.DataFrame(
        {
            "_row": oof_indices,
            "fold_id": oof_fold_ids,
            "y_pred_oof": oof_y_pred,
        },
        schema={
            "_row": pl.Int64,
            "fold_id": pl.Int64,
            "y_pred_oof": pl.Float64,
        },
    )
    enriched = (
        selected.with_row_index(name="_row")
        .with_columns(pl.col("_row").cast(pl.Int64))
        .join(oof_df, on="_row", how="left")
        .drop("_row")
        .rename({_TARGET_COLUMN: "y_true"})
        .sort("object_id")
    )
    buf = io.BytesIO()
    enriched.write_parquet(buf)
    return buf.getvalue()
