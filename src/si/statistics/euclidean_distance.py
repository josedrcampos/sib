import numpy as np


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum((x - y) ** 2, axis=1))
