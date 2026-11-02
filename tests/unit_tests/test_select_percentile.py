import os
from unittest import TestCase

import numpy as np

from datasets import DATASETS_PATH
from si.data.dataset import Dataset
from si.feature_selection.select_percentile import SelectPercentile
from si.io.csv_file import read_csv
from si.statistics.f_classification import f_classification


class TestSelectPercentile(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_fit(self):
        selector = SelectPercentile(score_func=f_classification, percentile=50)
        selector.fit(self.dataset)

        self.assertEqual(4, len(selector.F))
        self.assertEqual(4, len(selector.p))
        self.assertTrue(selector.is_fitted)

    def test_transform(self):
        selector = SelectPercentile(score_func=f_classification, percentile=50)
        new_dataset = selector.fit_transform(self.dataset)

        # 50% of 4 features = 2 features
        self.assertEqual((150, 2), new_dataset.shape())
        # the two petal features have the highest F values in iris
        self.assertEqual(['petal_length', 'petal_width'], new_dataset.features)
        self.assertEqual(self.dataset.y.tolist(), new_dataset.y.tolist())

    def test_ties_example_from_slides(self):
        F = np.array([1.2, 3.4, 2.1, 5.6, 4.3, 5.6, 7.8, 6.5, 5.6, 3.2])
        dataset = Dataset(np.random.rand(6, 10), np.array([0, 1, 0, 1, 0, 1]))

        selector = SelectPercentile(score_func=lambda ds: (F, np.zeros_like(F)), percentile=40)
        new_dataset = selector.fit_transform(dataset)

        # 40% of 10 = 4 features: the two above 5.6 plus the first two tied at 5.6
        self.assertEqual(4, new_dataset.shape()[1])
        self.assertEqual(['feat_3', 'feat_5', 'feat_6', 'feat_7'], new_dataset.features)

    def test_percentile_100_and_0(self):
        all_features = SelectPercentile(percentile=100).fit_transform(self.dataset)
        no_features = SelectPercentile(percentile=0).fit_transform(self.dataset)
        self.assertEqual(4, all_features.shape()[1])
        self.assertEqual(0, no_features.shape()[1])

    def test_invalid_percentile(self):
        with self.assertRaises(ValueError):
            SelectPercentile(percentile=150)
