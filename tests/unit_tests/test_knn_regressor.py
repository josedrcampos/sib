import os
from unittest import TestCase

import numpy as np

from datasets import DATASETS_PATH
from si.data.dataset import Dataset
from si.io.csv_file import read_csv
from si.metrics.rmse import rmse
from si.model_selection.split import train_test_split
from si.models.knn_regressor import KNNRegressor


class TestKNNRegressor(TestCase):

    def setUp(self):
        self.dataset = read_csv(os.path.join(DATASETS_PATH, 'cpu', 'cpu.csv'), features=True, label=True)

    def test_rmse(self):
        self.assertAlmostEqual(0.0, rmse(np.array([1, 2, 3]), np.array([1, 2, 3])))
        # errors [1, -1, 1, -1] -> mean of squares 1 -> rmse 1
        self.assertAlmostEqual(1.0, rmse(np.array([1, 2, 3, 4]), np.array([2, 1, 4, 3])))

    def test_fit(self):
        knn = KNNRegressor(k=3).fit(self.dataset)
        self.assertIs(knn.dataset, self.dataset)

    def test_predict_mean_of_neighbours(self):
        train = Dataset(np.array([[0.0], [1.0], [10.0]]), np.array([0.0, 2.0, 100.0]))
        test = Dataset(np.array([[0.4]]), np.array([1.0]))
        knn = KNNRegressor(k=2).fit(train)
        # the 2 nearest neighbours of 0.4 are 0 and 1 -> mean of y = (0 + 2) / 2
        self.assertEqual([1.0], knn.predict(test).tolist())

    def test_score(self):
        train, test = train_test_split(self.dataset, test_size=0.2, random_state=42)
        knn = KNNRegressor(k=3).fit(train)
        score = knn.score(test)
        self.assertEqual(test.shape()[0], knn.predict(test).shape[0])
        self.assertTrue(np.isfinite(score) and score >= 0)
