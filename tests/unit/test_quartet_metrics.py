"""Tests for kadastra.ml.quartet_metrics — pure stat helpers used by
the Black/Grey/White/Naive quartet (ADR-0016).

These functions take ``(y_true, y_pred)`` numpy arrays and return:

- ``spearman_corr`` — Spearman rank correlation coefficient.
- ``percentile_asymmetry`` — per-decile Δ medians + tail
  redistribution shares (frac overpredicted in bottom decile,
  frac underpredicted in top decile). Captures the «дешёвые
  становятся дороже / дорогие дешевле» tax-redistribution cost
  from info/grid-rationale.md §13.2.
- ``simplification_loss`` — Δ MAPE between two model results
  (Black − White, Black − Naive, …). Returned in **percentage
  points**, matching how MAPE is reported elsewhere in the project.
"""

from __future__ import annotations

import math

import numpy as np

from kadastra.ml.quartet_metrics import (
    percentile_asymmetry,
    simplification_loss_pp,
    spearman_corr,
    wape,
)


def test_spearman_corr_perfect_positive() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.5, 1.6, 2.5, 3.5, 9.0])
    # Both arrays are strictly increasing → ρ = 1.0 even though the
    # absolute values disagree.
    assert math.isclose(spearman_corr(y_true, y_pred), 1.0, abs_tol=1e-9)


def test_spearman_corr_perfect_negative() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    assert math.isclose(spearman_corr(y_true, y_pred), -1.0, abs_tol=1e-9)


def test_spearman_corr_robust_to_monotone_transform() -> None:
    """Spearman is invariant under any strict monotone transform of
    either side — that's its point. Squaring positives keeps order."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred_a = np.array([2.0, 4.0, 5.0, 7.0, 9.0])
    y_pred_b = y_pred_a**2
    assert math.isclose(
        spearman_corr(y_true, y_pred_a),
        spearman_corr(y_true, y_pred_b),
        abs_tol=1e-12,
    )


def test_percentile_asymmetry_unbiased_predictor_zero_skew() -> None:
    """If y_pred == y_true exactly, Δ medians are zero and the
    bottom-decile-overpredicted / top-decile-underpredicted shares
    are zero too."""
    rng = np.random.default_rng(0)
    y_true = rng.uniform(50_000, 200_000, size=1000)
    y_pred = y_true.copy()
    out = percentile_asymmetry(y_true, y_pred)
    for p in ("p10", "p25", "p50", "p75", "p90"):
        assert math.isclose(out[f"{p}_pred_minus_true"], 0.0, abs_tol=1e-9)
    assert out["frac_overpredicted_in_bottom_decile"] == 0.0
    assert out["frac_underpredicted_in_top_decile"] == 0.0


def test_percentile_asymmetry_compresses_tails() -> None:
    """A predictor that pulls all values toward the mean overshoots
    in the bottom decile and undershoots in the top decile — exactly
    the tax-redistribution cost the methodology warns about."""
    y_true = np.linspace(50_000, 200_000, num=1000)
    mean = float(np.mean(y_true))
    # Compressed predictor: 50% true + 50% mean → tails shrink.
    y_pred = 0.5 * y_true + 0.5 * mean
    out = percentile_asymmetry(y_true, y_pred)
    # Bottom of distribution: pred > true → overpredicted share high.
    assert out["frac_overpredicted_in_bottom_decile"] > 0.99
    # Top of distribution: pred < true → underpredicted share high.
    assert out["frac_underpredicted_in_top_decile"] > 0.99
    # p10 of pred is above p10 of true.
    assert out["p10_pred_minus_true"] > 0
    # p90 of pred is below p90 of true.
    assert out["p90_pred_minus_true"] < 0
    # Median is invariant under symmetric mean-pull.
    assert math.isclose(out["p50_pred_minus_true"], 0.0, abs_tol=1.0)


def test_simplification_loss_pp_reports_difference_in_percentage_points() -> None:
    """``simplification_loss_pp(black_mape, white_mape)`` returns the
    cost of using White Box instead of Black Box, expressed in pp.
    Black Box has lower MAPE (good) → loss is positive when going
    to White Box. Both MAPEs are in fractional units (0.10 == 10%).
    """
    black_mape = 0.0985  # 9.85%
    white_mape = 0.1230  # 12.30%
    loss = simplification_loss_pp(black_mape, white_mape)
    # White minus Black = 0.0245 → 2.45 pp.
    assert math.isclose(loss, 2.45, abs_tol=1e-6)


def test_simplification_loss_pp_can_be_negative() -> None:
    """If the simpler model accidentally beats the complex one (rare
    on tiny noisy classes), the loss is reported as negative."""
    black_mape = 0.50
    white_mape = 0.45
    loss = simplification_loss_pp(black_mape, white_mape)
    assert math.isclose(loss, -5.0, abs_tol=1e-6)


def test_wape_is_sum_normalized() -> None:
    """WAPE divides the total absolute error by the total target, so a
    small-target row doesn't dominate the metric the way MAPE would."""
    y_true = np.array([100_000.0, 200.0])
    y_pred = np.array([110_000.0, 300.0])
    # Total error = 10_000 + 100 = 10_100; total target = 100_200.
    assert math.isclose(wape(y_true, y_pred), 10_100.0 / 100_200.0, rel_tol=1e-12)


def test_wape_is_stable_when_single_row_is_tiny() -> None:
    """Contrast: MAPE on [100000, 200] with these errors is dominated by
    the tiny 200-row (50%), while WAPE stays ~10% because it's aggregate."""
    y_true = np.array([100_000.0, 200.0])
    y_pred = np.array([110_000.0, 300.0])
    mape = float(np.mean(np.abs(y_pred - y_true) / y_true))
    # MAPE: (10000/100000 + 100/200) / 2 = (0.10 + 0.50) / 2 = 0.30
    assert math.isclose(mape, 0.30, rel_tol=1e-12)
    # WAPE: 10100 / 100200 ≈ 0.1008 — stays near the 10% of the bulk.
    assert wape(y_true, y_pred) < 0.11


def test_wape_nan_on_zero_total() -> None:
    assert math.isnan(wape(np.array([0.0, 0.0]), np.array([1.0, 2.0])))
