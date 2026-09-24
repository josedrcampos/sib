import numpy as np

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    #outputs: float indicating the portion of well classified samples
    return np.sum(y_true == y_pred) / len(y_true)
    