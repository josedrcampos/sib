from typing import Callable

import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.rmse import rmse
from si.statistics.euclidean_distance import euclidean_distance


class KNNRegressor(Model):
    """
    KNN Regressor.
    Estimates the value of a sample as the mean of the values of its k nearest neighbours.
    """

    def __init__(self, k: int = 1, distance: Callable = euclidean_distance, **kwargs):
        """
        Parameters
        ----------
        k: int
            Number of nearest neighbours to consider
        distance: Callable
            Function that computes the distance between a sample and the training samples
        """
        super().__init__(**kwargs)
        self.k = k
        self.distance = distance

        # estimated parameters
        self.dataset = None

    def _fit(self, dataset: Dataset) -> 'KNNRegressor':
        """
        Stores the training dataset (KNN is a "lazy" model: there is nothing to estimate).
        """
        self.dataset = dataset
        return self

    def _get_closest_value(self, sample: np.ndarray) -> float:
        """
        Returns the mean value of the k nearest neighbours of a sample.
        """
        # 1. distance between the sample and every training sample
        distances = self.distance(sample, self.dataset.X)

        # 2. indexes of the k nearest neighbours
        k_nearest_neighbors = np.argsort(distances)[:self.k]

        # 3. values of y of those neighbours
        k_nearest_values = self.dataset.y[k_nearest_neighbors]

        # 4. mean of those values
        return np.mean(k_nearest_values)

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts the value of every sample in the dataset (step 5: apply to all samples).
        """
        return np.array([self._get_closest_value(sample) for sample in dataset.X])

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        RMSE between the real and predicted values.
        """
        return rmse(dataset.y, predictions)


if __name__ == '__main__':
    import os
    from datasets import DATASETS_PATH
    from si.io.csv_file import read_csv
    from si.model_selection.split import train_test_split

    cpu = read_csv(os.path.join(DATASETS_PATH, 'cpu', 'cpu.csv'), features=True, label=True)
    train, test = train_test_split(cpu, test_size=0.2, random_state=42)

    knn = KNNRegressor(k=3)
    knn.fit(train)
    print(f'RMSE on the test set: {knn.score(test):.4f}')
