import os
from unittest import TestCase

import numpy as np

from datasets import DATASETS_PATH
from si.io.csv_file import read_csv
from si.model_selection.split import train_test_split, stratified_train_test_split


class TestSplit(TestCase):

    def setUp(self):
        self.dataset = read_csv(os.path.join(DATASETS_PATH, 'iris', 'iris.csv'), features=True, label=True)

    def test_train_test_split(self):
        train, test = train_test_split(self.dataset, test_size=0.2, random_state=123)
        self.assertEqual(30, test.shape()[0])
        self.assertEqual(120, train.shape()[0])

    def test_stratified_sizes(self):
        train, test = stratified_train_test_split(self.dataset, test_size=0.2, random_state=123)
        self.assertEqual(30, test.shape()[0])
        self.assertEqual(120, train.shape()[0])

    def test_stratified_keeps_proportions(self):
        train, test = stratified_train_test_split(self.dataset, test_size=0.2, random_state=123)
        # iris has 50 samples per class -> 10 per class in test, 40 per class in train
        _, test_counts = np.unique(test.y, return_counts=True)
        _, train_counts = np.unique(train.y, return_counts=True)
        self.assertEqual([10, 10, 10], test_counts.tolist())
        self.assertEqual([40, 40, 40], train_counts.tolist())

    def test_stratified_uses_every_sample_once(self):
        train, test = stratified_train_test_split(self.dataset, test_size=0.2, random_state=123)
        # every sample ends up in exactly one of the two sets
        self.assertEqual(150, train.shape()[0] + test.shape()[0])

    def test_stratified_reproducible(self):
        _, test1 = stratified_train_test_split(self.dataset, random_state=7)
        _, test2 = stratified_train_test_split(self.dataset, random_state=7)
        self.assertTrue(np.array_equal(test1.X, test2.X))
