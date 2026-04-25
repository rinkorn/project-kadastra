import numpy as np
import polars as pl

from kadastra.ports.gold_feature_reader import GoldFeatureReaderPort
from kadastra.ports.gold_feature_store import GoldFeatureStorePort
from kadastra.ports.model_loader import ModelLoaderPort

_KEY_COLUMNS = ("h3_index", "resolution")


class InferValuation:
    def __init__(
        self,
        model_loader: ModelLoaderPort,
        gold_reader: GoldFeatureReaderPort,
        prediction_store: GoldFeatureStorePort,
        run_name_prefix: str,
    ) -> None:
        self._model_loader = model_loader
        self._gold_reader = gold_reader
        self._prediction_store = prediction_store
        self._run_name_prefix = run_name_prefix

    def execute(self, region_code: str, resolution: int, *, run_id: str | None = None) -> str:
        resolved_run_id = run_id or self._model_loader.find_latest_run_id(
            f"{self._run_name_prefix}{resolution}"
        )
        model = self._model_loader.load(resolved_run_id)

        gold = self._gold_reader.load(region_code, resolution)
        feature_cols = [c for c in gold.columns if c not in _KEY_COLUMNS]
        gold = gold.with_columns(
            [pl.col(c).fill_null(0).cast(pl.Float64) for c in feature_cols]
        )

        X = gold.select(feature_cols).to_numpy().astype(np.float64)
        preds = np.asarray(model.predict(X), dtype=np.float64)

        out = pl.DataFrame(
            {
                "h3_index": gold["h3_index"],
                "resolution": gold["resolution"],
                "predicted_value": preds,
            }
        )
        self._prediction_store.save(region_code, resolution, out)
        return resolved_run_id
