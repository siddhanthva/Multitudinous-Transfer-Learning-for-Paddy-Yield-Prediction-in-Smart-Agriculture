from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass
class RegressionMetrics:
    r2: float
    mse: float
    mae: float
    rmse: float
    acc_tol: float
    accuracy: float

    def to_dict(self) -> dict:
        return asdict(self)


def tolerance_accuracy(y_true: np.ndarray, y_pred: np.ndarray, tol: float) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have same shape.")
    return float(np.mean(np.abs(y_pred - y_true) <= tol))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, tol: float) -> RegressionMetrics:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return RegressionMetrics(
        r2=float(r2_score(y_true, y_pred)),
        mse=float(mean_squared_error(y_true, y_pred)),
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        acc_tol=float(tol),
        accuracy=tolerance_accuracy(y_true, y_pred, tol=tol),
    )

