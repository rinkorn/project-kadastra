import numpy as np
import polars as pl

from kadastra.domain.asset_class import AssetClass
from kadastra.ml.object_feature_matrix import build_object_feature_matrix
from kadastra.ports.model_loader import ModelLoaderPort
from kadastra.ports.valuation_object_reader import ValuationObjectReaderPort
from kadastra.ports.valuation_object_store import ValuationObjectStorePort

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


class InferObjectValuation:
    def __init__(
        self,
        model_loader: ModelLoaderPort,
        reader: ValuationObjectReaderPort,
        prediction_store: ValuationObjectStorePort,
        run_name_prefix: str,
    ) -> None:
        self._model_loader = model_loader
        self._reader = reader
        self._prediction_store = prediction_store
        self._run_name_prefix = run_name_prefix

    def execute(
        self,
        region_code: str,
        asset_class: AssetClass,
        *,
        run_id: str | None = None,
    ) -> str:
        resolved_run_id = run_id or self._model_loader.find_latest_run_id(
            f"{self._run_name_prefix}{asset_class.value}"
        )
        model = self._model_loader.load(resolved_run_id)

        df = self._reader.load(region_code, asset_class)

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
        X = build_object_feature_matrix(
            df, numeric_cols=numeric_cols, categorical_cols=categorical_cols
        )
        preds = np.asarray(model.predict(X), dtype=np.float64)

        out = pl.DataFrame(
            {
                "object_id": df["object_id"],
                "asset_class": df["asset_class"],
                "lat": df["lat"],
                "lon": df["lon"],
                "predicted_value": preds,
            }
        )
        self._prediction_store.save(region_code, asset_class, out)
        return resolved_run_id
