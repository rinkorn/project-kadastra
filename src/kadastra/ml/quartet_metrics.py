"""Stat helpers for the Black/Grey/White/Naive quartet (ADR-0016).

Three pure functions: Spearman rank correlation, percentile asymmetry
(per-decile Δ medians + tail redistribution shares), and simplification
loss in percentage points. Each takes ``(y_true, y_pred)`` numpy
arrays (or two MAPE scalars for ``simplification_loss_pp``) and
returns plain Python types — the use case marshals them into the
``quartet_metrics.json`` artifact.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def spearman_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation between two 1-D arrays."""
    result = spearmanr(y_true, y_pred)
    # SciPy returns SignificanceResult (or tuple-shaped fallback);
    # both expose ``.statistic`` at runtime, but the published stubs
    # don't surface it. Pull through __getattr__ to keep both runtime
    # behavior and pyright happy.
    return float(getattr(result, "statistic"))  # noqa: B009


def percentile_asymmetry(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """Per-decile bias + tail-redistribution shares.

    Returns a dict with:

    - ``p{10,25,50,75,90}_pred_minus_true`` — the corresponding
      percentile of ``y_pred`` minus the same percentile of
      ``y_true``. Negative on the upper tail + positive on the lower
      tail = mean-reverting predictor.
    - ``frac_overpredicted_in_bottom_decile`` — share of rows whose
      ``y_true`` is in the bottom 10 % AND ``y_pred > y_true``.
    - ``frac_underpredicted_in_top_decile`` — symmetric on the top.
    """
    out: dict[str, float] = {}
    for p in (10, 25, 50, 75, 90):
        out[f"p{p}_pred_minus_true"] = float(
            np.percentile(y_pred, p) - np.percentile(y_true, p)
        )

    bottom_threshold = float(np.percentile(y_true, 10))
    top_threshold = float(np.percentile(y_true, 90))
    bottom_mask = y_true <= bottom_threshold
    top_mask = y_true >= top_threshold

    if bottom_mask.any():
        out["frac_overpredicted_in_bottom_decile"] = float(
            np.mean(y_pred[bottom_mask] > y_true[bottom_mask])
        )
    else:
        out["frac_overpredicted_in_bottom_decile"] = 0.0

    if top_mask.any():
        out["frac_underpredicted_in_top_decile"] = float(
            np.mean(y_pred[top_mask] < y_true[top_mask])
        )
    else:
        out["frac_underpredicted_in_top_decile"] = 0.0

    return out


def simplification_loss_pp(black_mape: float, simpler_mape: float) -> float:
    """Cost of using the simpler model in percentage points.

    Both inputs are MAPE in fractional units (0.10 == 10 %); the
    return is in percentage points (2.45 == 2.45 pp). Positive when
    the simpler model is worse, which is the usual direction.
    """
    return (simpler_mape - black_mape) * 100.0
