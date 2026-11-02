import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root mean squared error: sqrt( (1/n) * sum((y_true - y_pred)^2) ).

    Same unit as y, so it can be read as "the typical size of the error".

    Parameters
    ----------
    y_true: np.ndarray
        The real values
    y_pred: np.ndarray
        The predicted values

    Returns
    -------
    rmse: float
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
