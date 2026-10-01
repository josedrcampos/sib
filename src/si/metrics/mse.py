import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean squared error: (1/n) * sum((y_true - y_pred)^2).

    Parameters
    ----------
    y_true: np.ndarray
        The real values
    y_pred: np.ndarray
        The predicted values

    Returns
    -------
    mse: float
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    return float(np.mean((y_true - y_pred) ** 2))
