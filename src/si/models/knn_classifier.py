from typing import Callable

import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.statistics.euclidean_distance import euclidean_distance


class KNNClassifier(Model):
    """
    KNN Classifier.
    Estimates the class of a sample based on the k most similar examples (nearest neighbours).
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

    def _fit(self, dataset: Dataset) -> 'KNNClassifier':
        """
        Stores the training dataset.
        """
        self.dataset = dataset
        return self

    def _get_closest_label(self, sample: np.ndarray):
        """
        Returns the most common class among the k nearest neighbours of a sample.
        """
        # 1. distance between the sample and every training sample
        distances = self.distance(sample, self.dataset.X)

        # 2. indexes of the k nearest neighbours
        k_nearest_neighbors = np.argsort(distances)[:self.k]

        # 3. classes of those neighbours
        k_nearest_neighbors_labels = self.dataset.y[k_nearest_neighbors]

        # 4. most frequent class
        labels, counts = np.unique(k_nearest_neighbors_labels, return_counts=True)
        return labels[np.argmax(counts)]

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts the class of every sample in the dataset.
        """
        # 5. apply to every test sample (list comprehension avoids string truncation)
        return np.array([self._get_closest_label(sample) for sample in dataset.X])

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Accuracy between the real and predicted classes.
        """
        return accuracy(dataset.y, predictions)


if __name__ == '__main__':
    from si.model_selection.split import train_test_split

    dataset_ = Dataset.from_random(600, 100, 2)
    dataset_train, dataset_test = train_test_split(dataset_, test_size=0.2)

    knn = KNNClassifier(k=3)
    knn.fit(dataset_train)

    score = knn.score(dataset_test)
    print(f'The accuracy of the model is: {score}')