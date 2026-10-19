from typing import Tuple

import numpy as np
from scipy import stats

from si.data.dataset import Dataset


def f_classification(dataset: Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scoring function for classification problems.
    Computes the one-way ANOVA F-value and p-value for every feature.

    Parameters
    ----------
    dataset: Dataset
        A labeled dataset

    Returns
    -------
    F: np.ndarray, shape (n_features,)
        F values
    p: np.ndarray, shape (n_features,)
        p values
    """
    classes = dataset.get_classes()
    groups = [dataset.X[dataset.y == c] for c in classes]
    F, p = stats.f_oneway(*groups)
    return F, p


if __name__ == '__main__':
    X = np.array([[1, 2, 3], [1, 3, 4], [5, 6, 7], [6, 7, 8]])
    y = np.array([0, 0, 1, 1])
    dataset = Dataset(X, y)
    F, p = f_classification(dataset)
    print("F:", F)
    print("p:", p)