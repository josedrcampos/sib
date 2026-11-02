import os
from unittest import TestCase

from datasets import DATASETS_PATH
from si.feature_selection.select_k_best import SelectKBest
from si.io.csv_file import read_csv


class TestSelectKBest(TestCase):

    def setUp(self):
        self.dataset = read_csv(os.path.join(DATASETS_PATH, 'iris', 'iris.csv'), features=True, label=True)

    def test_fit_transform(self):
        selector = SelectKBest(k=2)
        new_dataset = selector.fit_transform(self.dataset)

        self.assertEqual(4, len(selector.F))
        self.assertEqual((150, 2), new_dataset.shape())
        self.assertEqual(['petal_length', 'petal_width'], new_dataset.features)

    def test_k_larger_than_n_features(self):
        new_dataset = SelectKBest(k=10).fit_transform(self.dataset)
        self.assertEqual(4, new_dataset.shape()[1])
