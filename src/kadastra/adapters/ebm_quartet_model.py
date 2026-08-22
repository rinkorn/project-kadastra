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
        interactions_exclude: tuple[str, ...] = (),
    ) -> None:
        self._max_bins = max_bins
        self._interactions = interactions
        # When TrainQuartet runs folds in parallel, callers pass
        # n_jobs=1 here so EBM's outer_bags stay sequential — a parallel
        # outer × parallel inner combination oversubscribes the machine.
        self._n_jobs = n_jobs
        # Feature names banned from PAIRS only (their main effects stay).
        # High-cardinality categoricals (vri ~13k values, kadnum_quarter
        # ~3.6k) make the interaction tensor explode (64 bins × 13k
        # categories) — hours per pair — and overfit anyway (~12 rows
        # per category). interpret-ml's ``exclude`` drops a bare name's
        # MAIN term too, so we exclude every pair containing the name.
        self._interactions_exclude = interactions_exclude
        self._model: ExplainableBoostingRegressor | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        cat_feature_indices: list[int] | None = None,
        feature_names: list[str] | None = None,
    ) -> None:
        cat_set = set(cat_feature_indices or [])
        feature_types = ["nominal" if i in cat_set else "continuous" for i in range(X.shape[1])]
        # Default EBM n_jobs is -2 ("all-1 cores"). When the caller pins
        # us to 1 (parallel-folds outer loop), pass it through.
        n_jobs = self._n_jobs if self._n_jobs is not None else -2
        exclude: list[tuple[str, str]] | None = None
        if self._interactions_exclude:
            if feature_names is None:
                raise ValueError("feature_names are required when interactions_exclude is set")
            banned = set(self._interactions_exclude)
            missing = banned - set(feature_names)
            if missing:
                raise ValueError(f"interactions_exclude names not in feature_names: {sorted(missing)}")
            exclude = [(name, other) for name in banned for other in feature_names if other != name]
        model = ExplainableBoostingRegressor(
            feature_names=feature_names,
            feature_types=feature_types,
            max_bins=self._max_bins,
            interactions=self._interactions,
            exclude=exclude,
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
        if self._model is None:
            raise RuntimeError("EbmQuartetModel.eval_terms before fit")
        terms = self._model.eval_terms(X)
        return np.asarray(terms, dtype=np.float64)

    def intercept(self) -> float:
        """Global intercept of the fitted EBM."""
        if self._model is None:
            raise RuntimeError("EbmQuartetModel.intercept before fit")
        return float(np.asarray(self._model.intercept_, dtype=np.float64).ravel()[0])

    def term_feature_indices(self) -> list[tuple[int, ...]]:
        """Feature indices per term (length-1 for mains, 2 for interactions)."""
        if self._model is None:
            raise RuntimeError("EbmQuartetModel.term_feature_indices before fit")
        return [tuple(int(i) for i in idxs) for idxs in self._model.term_features_]

    def serialize(self) -> bytes:
        if self._model is None:
            raise RuntimeError("EbmQuartetModel.serialize before fit")
        return pickle.dumps((self._max_bins, self._interactions, self._interactions_exclude, self._model))

    @classmethod
    def deserialize(cls, blob: bytes) -> EbmQuartetModel:
        payload = pickle.loads(blob)
        # Backward compat: pre-interactions_exclude artifacts pickled a
        # 3-tuple (max_bins, interactions, model).
        if len(payload) == 3:
            max_bins, interactions, model = payload
            interactions_exclude: tuple[str, ...] = ()
        else:
            max_bins, interactions, interactions_exclude, model = payload
        instance = cls(
            max_bins=max_bins,
            interactions=interactions,
            interactions_exclude=interactions_exclude,
        )
        instance._model = model
        return instance
