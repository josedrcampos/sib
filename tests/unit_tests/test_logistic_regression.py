import os
from unittest import TestCase

import numpy as np

from datasets import DATASETS_PATH
from si.io.data_file import read_data_file
from si.model_selection.split import stratified_train_test_split
from si.models.logistic_regression import LogisticRegression


class TestLogisticRegression(TestCase):

    def setUp(self):
        # breast-bin.csv has no header row -> read_data_file
        self.dataset = read_data_file(os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv'),
                                      sep=',', label=True)
        self.train, self.test = stratified_train_test_split(self.dataset, test_size=0.2, random_state=42)

    def test_fit(self):
        model = LogisticRegression(l2_penalty=1.0, alpha=0.001, max_iter=2000).fit(self.train)
        self.assertEqual(self.train.X.shape[1], model.theta.shape[0])
        self.assertIsNotNone(model.theta_zero)
        self.assertGreater(len(model.cost_history), 1)
        costs = list(model.cost_history.values())
        self.assertLess(costs[-1], costs[0])

    def test_predict_is_binary(self):
        model = LogisticRegression(alpha=0.001, max_iter=2000).fit(self.train)
        predictions = model.predict(self.test)
        self.assertEqual(self.test.shape()[0], predictions.shape[0])
        self.assertTrue(set(np.unique(predictions)) <= {0, 1})

    def test_score_and_cost(self):
        model = LogisticRegression(alpha=0.001, max_iter=2000).fit(self.train)
        self.assertGreater(model.score(self.test), 0.9)
        cost = model.cost(self.test)
        self.assertTrue(np.isfinite(cost) and cost > 0)
