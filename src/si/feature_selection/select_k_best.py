from typing import Callable

import numpy as np

from si.base.transformer import Transformer
from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification


class SelectKBest(Transformer):
    """
    Selects the k features with the highest score (F value) given by score_func.
    """

    def __init__(self, score_func: Callable = f_classification, k: int = 10, **kwargs):
        """
        Parameters
        ----------
        score_func: Callable
            Function that receives a Dataset and returns (F, p) for every feature.
        k: int
            Number of features to select.
        """
        super().__init__(**kwargs)
        if k <= 0:
            raise ValueError("k must be a positive integer")
        self.score_func = score_func
        self.k = k

        # estimated parameters
        self.F = None
        self.p = None

    def _fit(self, dataset: Dataset) -> 'SelectKBest':
        """
        Estimates the F and p values of every feature with score_func.
        """
        self.F, self.p = self.score_func(dataset)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Keeps the k features with the highest F values.
        """
        k = min(self.k, len(self.F))
        # constant features give F = NaN; F is always >= 0, so -1 makes them the worst score
        scores = np.where(np.isnan(self.F), -1.0, self.F)
        # argsort is ascending, so the last k indexes are the k highest F values;
        # sorting them again keeps the original column order
        idxs = np.sort(np.argsort(scores)[-k:])

        features = np.array(dataset.features)[idxs].tolist()
        return Dataset(X=dataset.X[:, idxs], y=dataset.y, features=features, label=dataset.label)


if __name__ == '__main__':
    dataset = Dataset(X=np.array([[0, 2, 0, 3],
                                  [0, 1, 4, 3],
                                  [0, 1, 1, 3]]),
                      y=np.array([0, 1, 0]),
                      features=["f1", "f2", "f3", "f4"],
                      label="y")

    selector = SelectKBest(k=2)
    new_dataset = selector.fit_transform(dataset)
    print(new_dataset.features)
