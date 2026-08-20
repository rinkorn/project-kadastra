"""Unit tests for distribution-comparison metrics (эпик 001, этап 5)."""

import numpy as np
import polars as pl
import pytest

from kadastra.etl.representativeness import (
    compare_distributions,
    compare_feature,
    compute_psi,
    psi_verdict,
)


def test_psi_identical_distributions_is_near_zero() -> None:
    rng = np.random.default_rng(42)
    population = rng.normal(100.0, 15.0, size=10_000)
    sample = rng.normal(100.0, 15.0, size=2_000)
    assert compute_psi(population, sample) < 0.1


def test_psi_shifted_distribution_exceeds_shift_threshold() -> None:
    rng = np.random.default_rng(42)
    population = rng.normal(100.0, 15.0, size=10_000)
    sample = rng.normal(160.0, 15.0, size=2_000)
    assert compute_psi(population, sample) > 0.25


def test_psi_constant_population_is_nan() -> None:
    assert np.isnan(compute_psi(np.full(100, 5.0), np.full(50, 5.0)))


def test_psi_ignores_nan_and_inf() -> None:
    rng = np.random.default_rng(42)
    population = rng.normal(0.0, 1.0, size=1_000)
    sample = rng.normal(0.0, 1.0, size=500)
    population[::10] = np.nan
    sample[::7] = np.inf
    assert compute_psi(population, sample) < 0.1


def test_compare_feature_identical_has_zero_ks() -> None:
    values = np.linspace(0.0, 1.0, 1_000)
    metrics = compare_feature(values, values)
    assert metrics["ks_stat"] == pytest.approx(0.0)
    assert metrics["psi"] == pytest.approx(0.0, abs=1e-6)


def test_compare_feature_empty_side_is_nan() -> None:
    metrics = compare_feature(np.array([1.0, 2.0]), np.array([]))
    assert np.isnan(metrics["ks_stat"])
    assert np.isnan(metrics["psi"])


def test_psi_verdict_thresholds() -> None:
    assert psi_verdict(0.05) == "ok"
    assert psi_verdict(0.15) == "moderate"
    assert psi_verdict(0.30) == "shift"
    assert psi_verdict(float("nan")) == "n/a"


def test_compare_distributions_rows_and_coverage() -> None:
    grid = pl.DataFrame(
        {
            "h3_index": ["a", "b", "c", "d"],
            "f1": [0.0, 1.0, 2.0, 3.0],
            "count_x": [0, 1, 2, 3],  # Int64 counter — must cast cleanly
        },
        schema={"h3_index": pl.String, "f1": pl.Float64, "count_x": pl.Int64},
    )
    # Sample keeps duplicate cells on purpose (sample weighting).
    sample = pl.DataFrame(
        {
            "h3_index": ["a", "a", "b"],
            "f1": [0.0, 0.0, 1.0],
            "count_x": [0, 0, 1],
        },
        schema={"h3_index": pl.String, "f1": pl.Float64, "count_x": pl.Int64},
    )
    report = compare_distributions(grid, sample, {"f1": "geom_distance", "count_x": "zonal"}, "overall")

    assert report.height == 2
    assert set(report["feature"]) == {"f1", "count_x"}
    row = report.filter(pl.col("feature") == "f1").row(0, named=True)
    assert row["feature_set"] == "geom_distance"
    assert row["segment"] == "overall"
    assert row["n_grid"] == 4
    assert row["n_sample"] == 3
    assert row["n_sample_cells"] == 2
    assert row["coverage"] == pytest.approx(0.5)
    assert row["verdict"] in {"ok", "moderate", "shift"}


def test_compare_distributions_skips_missing_columns() -> None:
    grid = pl.DataFrame({"h3_index": ["a"], "f1": [1.0]})
    sample = pl.DataFrame({"h3_index": ["a"], "other": [1.0]})
    report = compare_distributions(grid, sample, {"f1": "set", "other": "set"}, "s")
    assert report.is_empty()
