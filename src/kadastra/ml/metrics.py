import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: y_true={y_true.shape} vs y_pred={y_pred.shape}")

    errors = y_pred - y_true
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))

    nonzero = y_true != 0.0
    mape = (
        float("nan")
        if not np.any(nonzero)
        else float(np.mean(np.abs(errors[nonzero] / y_true[nonzero])))
    )

    return {"mae": mae, "rmse": rmse, "mape": mape}
