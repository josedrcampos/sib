from typing import Callable, Union

import numpy as np

from si.base.model import Model
from si.base.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.statistics.euclidean_distance import euclidean_distance


class KNNClassifier(Model):
    def __init__(self, k: int = 5, distance=euclidean_distance, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.distance = distance

        # estimated parameters
        self.dataset = None

    def _fit(self, dataset: Dataset):
        self.dataset = dataset
        return self

    def _get_closest_label(self, sample: np.ndarray):
        distances = self.distance(sample, self.dataset.X)
        
        #todos os vizinhos da k distância mais próxima (?) - escolher os 3 vizinhos mais próximos, pomos k=3
        k_nearest_neighbors = np.argsort(distances)[: self.k]

        k_nearest_neighbors_labels = self.dataset.y[k_nearest_neighbors]
        #queremos também saber o que são esses vizinhos, o que está nessas distâncias mais próximas
        labels, counts = np.unique(k_nearest_neighbors_labels, return_counts=True)
        return labels[np.argmax(counts)]

    def _predict(self, dataset):
        return np.apply_along_axis(self._get_closest_label, axis=1, arr=dataset.X)

    def _score(self, dataset):
        predictions = self.predict(dataset)
        return accuracy(dataset.y, predictions)
    
if __name__ == '__main__':
    #import dataset
    from si.data.dataset import Dataset
    from si.model_selection.split import train_test_split
    
    #load and split the dataseti
    dataset_ = Dataset.from_random(600, q00, 2)
    dataset_train, dataset_test = train_test_split(dataset__, test_size=0.2)
    
    #initialize the KNN classifier
    knn = KNNClassifier(k=3)
    
    # fit the model to the train dataset
    knn.fit(dataset_train)
    
    #evaluate the model on the test dataset
    score = knn.score(dataset_test)
    print(f'The accuracy of the model is: {score}.')
    
    
    '''
    
    '''
    