import math

import numpy as np
import pytest

from kadastra.ml.metrics import regression_metrics


def test_perfect_prediction_gives_zero_errors() -> None:
    y = np.array([100.0, 200.0, 300.0])

    m = regression_metrics(y, y.copy())

    assert m["mae"] == 0.0
    assert m["rmse"] == 0.0
    assert m["mape"] == 0.0


def test_mae_is_mean_absolute_error() -> None:
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 180.0, 330.0])

    m = regression_metrics(y_true, y_pred)

    # |10| + |20| + |30| = 60 / 3 = 20
    assert m["mae"] == pytest.approx(20.0)


def test_rmse_is_root_mean_squared_error() -> None:
    y_true = np.array([0.0, 0.0, 0.0])
    y_pred = np.array([1.0, 2.0, 3.0])

    m = regression_metrics(y_true, y_pred)

    # sqrt((1+4+9)/3) = sqrt(14/3)
    assert m["rmse"] == pytest.approx(math.sqrt(14 / 3))


def test_mape_is_mean_absolute_percentage_error() -> None:
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([110.0, 180.0])

    m = regression_metrics(y_true, y_pred)

    # (|10|/100 + |20|/200) / 2 = (0.1 + 0.1) / 2 = 0.1
    assert m["mape"] == pytest.approx(0.1)


def test_mape_skips_zero_targets() -> None:
    y_true = np.array([0.0, 100.0, 0.0, 200.0])
    y_pred = np.array([5.0, 110.0, 50.0, 180.0])

    m = regression_metrics(y_true, y_pred)

    # Only the nonzero entries: (10/100 + 20/200) / 2 = 0.1
    assert m["mape"] == pytest.approx(0.1)


def test_mape_returns_nan_when_all_targets_zero() -> None:
    y_true = np.array([0.0, 0.0, 0.0])
    y_pred = np.array([1.0, 2.0, 3.0])

    m = regression_metrics(y_true, y_pred)

    assert math.isnan(m["mape"])


def test_raises_on_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape"):
        regression_metrics(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))
