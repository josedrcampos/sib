import os
from unittest import TestCase

import numpy as np

from datasets import DATASETS_PATH
from si.data.dataset import Dataset
from si.io.csv_file import read_csv
from si.model_selection.split import train_test_split
from si.models.ridge_regression_least_squares import RidgeRegressionLeastSquares


class TestRidgeRegressionLeastSquares(TestCase):

    def setUp(self):
        self.dataset = read_csv(os.path.join(DATASETS_PATH, 'cpu', 'cpu.csv'), features=True, label=True)
        self.train, self.test = train_test_split(self.dataset, test_size=0.2, random_state=42)

    def test_fit(self):
        model = RidgeRegressionLeastSquares(l2_penalty=1.0, scale=True).fit(self.train)
        self.assertEqual(self.train.X.shape[1], model.theta.shape[0])
        self.assertIsNotNone(model.theta_zero)

    def test_recovers_exact_line_without_penalty(self):
        # y = 3 + 2*x1 - x2, no noise: with lambda = 0 the solution must be exact
        X = np.random.default_rng(0).random((50, 2))
        y = 3 + 2 * X[:, 0] - X[:, 1]
        model = RidgeRegressionLeastSquares(l2_penalty=0.0, scale=False).fit(Dataset(X, y))
        np.testing.assert_allclose(model.theta_zero, 3, atol=1e-8)
        np.testing.assert_allclose(model.theta, [2, -1], atol=1e-8)

    def test_score(self):
        model = RidgeRegressionLeastSquares(l2_penalty=1.0, scale=True).fit(self.train)
        score = model.score(self.test)
        self.assertTrue(np.isfinite(score) and score >= 0)
