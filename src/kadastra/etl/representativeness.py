"""Distribution-comparison metrics for the representativeness report.

Methodology §2 п.5 (grid-rationale.md): a training sample is
representative when the distribution of each ЦОФ over the sample
matches the distribution over the full territory (the Слой 1 grid).
This module implements the two metrics used for that comparison:

- **PSI** (population stability index) — decile bins of the population
  (grid) distribution; the industry verdict thresholds are
  ``<0.1`` ok, ``0.1–0.25`` moderate, ``>=0.25`` shift.
- **KS statistic** (``scipy.stats.ks_2samp``) — effect size; the
  p-value is nominal at N~1e5 and reported for completeness only.

Pure functions over numpy/polars — no ports, no I/O.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats

PSI_MODERATE_THRESHOLD = 0.1
PSI_SHIFT_THRESHOLD = 0.25
# Replaced zero bin shares so a bin empty on one side does not blow up
# the log-ratio; small enough not to mask a real shift.
_PSI_EPSILON = 1e-4


def compute_psi(population: np.ndarray, sample: np.ndarray, *, bins: int = 10) -> float:
    """Population stability index of ``sample`` against ``population``.

    Bin edges are the deciles of the population distribution with open
    ends, so sample values outside the population range land in the
    first/last bin. Returns ``nan`` when the population is constant
    (deciles degenerate and PSI is undefined).
    """
    population = _clean(population)
    sample = _clean(sample)
    if population.size == 0 or sample.size == 0:
        return float("nan")
    if np.unique(population).size < 2:
        # Constant population: deciles degenerate, PSI is undefined.
        return float("nan")
    quantiles = np.quantile(population, np.linspace(0.0, 1.0, bins + 1))
    inner = np.unique(quantiles[1:-1])
    if inner.size == 0:
        return float("nan")
    edges = np.concatenate(([-np.inf], inner, [np.inf]))
    pop_share = np.histogram(population, bins=edges)[0] / population.size
    smp_share = np.histogram(sample, bins=edges)[0] / sample.size
    pop_share = np.clip(pop_share, _PSI_EPSILON, None)
    smp_share = np.clip(smp_share, _PSI_EPSILON, None)
    return float(np.sum((pop_share - smp_share) * np.log(pop_share / smp_share)))


def psi_verdict(psi: float) -> str:
    """Industry PSI thresholds: ok / moderate / shift (``nan`` → ``n/a``)."""
    if np.isnan(psi):
        return "n/a"
    if psi < PSI_MODERATE_THRESHOLD:
        return "ok"
    if psi < PSI_SHIFT_THRESHOLD:
        return "moderate"
    return "shift"


def compare_feature(population: np.ndarray, sample: np.ndarray) -> dict[str, float]:
    """KS + PSI for one feature column; ``nan``-safe (NaN/inf dropped)."""
    population = _clean(population)
    sample = _clean(sample)
    if population.size == 0 or sample.size == 0:
        return {"ks_stat": float("nan"), "ks_pvalue": float("nan"), "psi": float("nan")}
    ks = stats.ks_2samp(population, sample)
    # SciPy returns KstestResult; the published stubs don't surface its
    # fields — same getattr workaround as quartet_metrics.spearman_corr.
    return {
        "ks_stat": float(getattr(ks, "statistic")),  # noqa: B009
        "ks_pvalue": float(getattr(ks, "pvalue")),  # noqa: B009
        "psi": compute_psi(population, sample),
    }


def compare_distributions(
    grid: pl.DataFrame,
    sample: pl.DataFrame,
    feature_sets: dict[str, str],
    segment: str,
) -> pl.DataFrame:
    """Long-format comparison rows for every shared feature column.

    ``grid`` is the full-territory Слой 1 frame (population), ``sample``
    is the grid joined to the training objects (one row per object —
    duplicate cells are kept on purpose: that is the sample weighting).
    ``feature_sets`` maps feature column name → Слой 1 feature set.
    """
    rows: list[dict[str, object]] = []
    n_grid_cells = grid.height
    n_sample_cells = sample["h3_index"].n_unique() if sample.height else 0
    coverage = n_sample_cells / n_grid_cells if n_grid_cells else float("nan")
    for feature, feature_set in feature_sets.items():
        if feature not in grid.columns or feature not in sample.columns:
            continue
        metrics = compare_feature(
            grid[feature].cast(pl.Float64).to_numpy(),
            sample[feature].cast(pl.Float64).to_numpy(),
        )
        rows.append(
            {
                "feature_set": feature_set,
                "feature": feature,
                "segment": segment,
                "n_grid": n_grid_cells,
                "n_sample": sample.height,
                "n_sample_cells": n_sample_cells,
                "coverage": coverage,
                "ks_stat": metrics["ks_stat"],
                "ks_pvalue": metrics["ks_pvalue"],
                "psi": metrics["psi"],
                "verdict": psi_verdict(metrics["psi"]),
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "feature_set": pl.String,
            "feature": pl.String,
            "segment": pl.String,
            "n_grid": pl.Int64,
            "n_sample": pl.Int64,
            "n_sample_cells": pl.Int64,
            "coverage": pl.Float64,
            "ks_stat": pl.Float64,
            "ks_pvalue": pl.Float64,
            "psi": pl.Float64,
            "verdict": pl.String,
        },
    )


def _clean(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64, copy=False)
    return values[np.isfinite(values)]
