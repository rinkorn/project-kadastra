"""TrainQuartet use case (ADR-0016).

Wires together the four QuartetModel adapters (Black/White/Grey/Naive)
and computes per-fold + aggregate metrics on a single set of spatial-CV
folds, so per-fold comparison is honest. Logs everything as a single
``quartet-object-{class}_<ts>`` run via ModelRegistryPort, with
``quartet_metrics.json`` and per-model OOF parquets as artifacts.

The Black Box's final fit is what's stored as the run's primary
``model``; the other three are persisted as raw bytes inside
``artifacts``.
"""

from __future__ import annotations

import io
import json

import h3
import numpy as np
import polars as pl

from kadastra.adapters.catboost_quartet_model import CatBoostQuartetModel
from kadastra.adapters.ebm_quartet_model import EbmQuartetModel
from kadastra.adapters.grey_tree_quartet_model import GreyTreeQuartetModel
from kadastra.adapters.naive_linear_quartet_model import NaiveLinearQuartetModel
from kadastra.domain.asset_class import AssetClass
from kadastra.ml.metrics import regression_metrics
from kadastra.ml.object_feature_columns import select_object_feature_columns
from kadastra.ml.object_feature_matrix import build_object_feature_matrix
from kadastra.ml.quartet_metrics import (
    percentile_asymmetry,
    simplification_loss_pp,
    spearman_corr,
)
from kadastra.ml.spatial_kfold import spatial_kfold_split
from kadastra.ml.train import CatBoostParams
from kadastra.ports.model_registry import ModelRegistryPort
from kadastra.ports.valuation_object_reader import ValuationObjectReaderPort

_TARGET_COLUMN = "synthetic_target_rub_per_m2"
_NAIVE_NUMERIC = ("lat", "lon", "area_m2", "levels", "flats", "year_built")
_NAIVE_CATEGORICAL = ("asset_class",)


class TrainQuartet:
    def __init__(
        self,
        reader: ValuationObjectReaderPort,
        model_registry: ModelRegistryPort,
        *,
        catboost_params: CatBoostParams,
        ebm_max_bins: int,
        ebm_interactions: int,
        grey_tree_max_depth: int,
        n_splits: int,
        parent_resolution: int,
    ) -> None:
        self._reader = reader
        self._model_registry = model_registry
        self._catboost_params = catboost_params
        self._ebm_max_bins = ebm_max_bins
        self._ebm_interactions = ebm_interactions
        self._grey_tree_max_depth = grey_tree_max_depth
        self._n_splits = n_splits
        self._parent_resolution = parent_resolution

    def execute(self, region_code: str, asset_class: AssetClass) -> str:
        raise NotImplementedError
