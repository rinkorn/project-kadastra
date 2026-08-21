"""White Box adapter for the ADR-0016 quartet — interpret-ml's
ExplainableBoostingRegressor.

EBM is an additive model over per-feature shape functions + a small
number of pairwise interactions. Categorical columns are declared via
``feature_types``: ``'continuous'`` for numerics, ``'nominal'`` for
categoricals. The binner handles NaN in numerics and unseen-in-fit
categories without explicit imputation/encoding.
"""

from __future__ import annotations

import pickle
from typing import Any, cast

import numpy as np
from interpret.glassbox import ExplainableBoostingRegressor


class EbmQuartetModel:
    def __init__(
        self,
        *,
        max_bins: int = 256,
        interactions: int = 10,
        n_jobs: int | None = None,
    ) -> None:
        self._max_bins = max_bins
        self._interactions = interactions
        # When TrainQuartet runs folds in parallel, callers pass
        # n_jobs=1 here so EBM's outer_bags stay sequential — a parallel
        # outer × parallel inner combination oversubscribes the machine.
        self._n_jobs = n_jobs
        self._model: ExplainableBoostingRegressor | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        cat_feature_indices: list[int] | None = None,
    ) -> None:
        cat_set = set(cat_feature_indices or [])
        feature_types = ["nominal" if i in cat_set else "continuous" for i in range(X.shape[1])]
        # Default EBM n_jobs is -2 ("all-1 cores"). When the caller pins
        # us to 1 (parallel-folds outer loop), pass it through.
        n_jobs = self._n_jobs if self._n_jobs is not None else -2
        model = ExplainableBoostingRegressor(
            feature_types=feature_types,
            max_bins=self._max_bins,
            interactions=self._interactions,
            n_jobs=n_jobs,
        )
        model.fit(X, y)
        self._model = model

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("EbmQuartetModel.predict before fit")
        preds = self._model.predict(X)
        return np.asarray(preds, dtype=np.float64)

    def explain(self, X: np.ndarray, feature_names: list[str]) -> dict[str, Any]:
        """Per-sample feature contributions via interpret-ml ``explain_local``.

        Returns ``{"intercept": float, "terms": [{"feature", "value",
        "contribution"}]}``. The model was fit on a raw NumPy matrix with no
        feature names, so term indices are mapped back through ``feature_names``
        (numeric then categorical — the same order ``build_object_feature_matrix``
        emits). ``intercept + sum(contribution) == predict(X)``.
        """
        if self._model is None:
            raise RuntimeError("EbmQuartetModel.explain before fit")
        raw = self._model.explain_local(X).data(key=0)
        if raw is None:
            raise RuntimeError("EBM explain_local returned no data for key=0")
        data = cast(dict[str, Any], raw)
        intercept = float(data["extra"]["scores"][0])
        terms: list[dict[str, Any]] = []
        for idx, (score, value) in enumerate(zip(data["scores"], data["values"], strict=True)):
            indices = self._model.term_features_[idx]
            name = " & ".join(feature_names[i] for i in indices)
            terms.append({"feature": name, "value": value, "contribution": float(score)})
        return {"intercept": intercept, "terms": terms}

    def eval_terms(self, X: np.ndarray) -> np.ndarray:
        """Per-sample, per-term contributions — (n_samples, n_terms).

        ``intercept() + eval_terms(X).sum(axis=1) == predict(X)``.
        """
        raise NotImplementedError

    def intercept(self) -> float:
        """Global intercept of the fitted EBM."""
        raise NotImplementedError

    def term_feature_indices(self) -> list[tuple[int, ...]]:
        """Feature indices per term (length-1 for mains, 2 for interactions)."""
        raise NotImplementedError

    def serialize(self) -> bytes:
        if self._model is None:
            raise RuntimeError("EbmQuartetModel.serialize before fit")
        return pickle.dumps((self._max_bins, self._interactions, self._model))

    @classmethod
    def deserialize(cls, blob: bytes) -> EbmQuartetModel:
        max_bins, interactions, model = pickle.loads(blob)
        instance = cls(max_bins=max_bins, interactions=interactions)
        instance._model = model
        return instance
