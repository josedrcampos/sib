import numpy as np

from si.base.model import Model
from si.metrics.accuracy import accuracy
from si.statistics.euclidean_distance import euclidean_distance


class KNNClassifier(Model):
    def __init__(self, k: int = 5, distance=euclidean_distance, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.distance = distance

        # estimated parameters
        self.dataset = None

    def _fit(self, dataset):
        self.dataset = dataset
        return self

    def _get_closest_label(self, sample: np.ndarray):
        distances = self.distance(sample, self.dataset.X)

        k_nearest_neighbors = np.argsort(distances)[: self.k]

        k_nearest_neighbors_labels = self.dataset.y[k_nearest_neighbors]

        labels, counts = np.unique(k_nearest_neighbors_labels, return_counts=True)
        return labels[np.argmax(counts)]

    def _predict(self, dataset):
        return np.apply_along_axis(self._get_closest_label, axis=1, arr=dataset.X)

    def _score(self, dataset):
        predictions = self.predict(dataset)
        return accuracy(dataset.y, predictions)