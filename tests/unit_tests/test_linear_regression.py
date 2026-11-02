import os
import unittest

import numpy as np

from datasets import DATASETS_PATH
from si.io.csv_file import read_csv
from si.model_selection.split import train_test_split
from si.models.linear_regression import RidgeRegression

class TestRidgeRegression(unittest.TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, "cpu", "cpu.csv")
        self.dataset = read_csv(filename=self.csv_file, sep=",", features=True, label=True)
        self.train, self.test = train_test_split(self.dataset, test_size=0.2, random_state=42)

    def test_fit(self):
        model = RidgeRegression(l2_penalty=1.0, alpha=0.01, max_iter=2000, patience=5, scale=True)
        model.fit(self.train)

        self.assertEqual(model.theta.shape[0], self.train.X.shape[1])
        self.assertIsNotNone(model.theta_zero)
        self.assertEqual(model.mean.shape[0], self.train.X.shape[1])
        self.assertEqual(model.std.shape[0], self.train.X.shape[1])
        # gradient descent: one cost per iteration, never more than max_iter
        self.assertGreater(len(model.cost_history), 1)
        self.assertLessEqual(len(model.cost_history), model.max_iter)

    def test_cost_decreases(self):
        model = RidgeRegression(l2_penalty=1.0, alpha=0.01, max_iter=2000, patience=5, scale=True)
        model.fit(self.train)
        costs = list(model.cost_history.values())
        self.assertLess(costs[-1], costs[0])

    def test_close_to_least_squares(self):
        # with enough iterations, gradient descent approaches the analytical solution
        from si.models.ridge_regression_least_squares import RidgeRegressionLeastSquares
        gd = RidgeRegression(l2_penalty=1.0, alpha=0.1, max_iter=20000, patience=5, scale=True).fit(self.train)
        ls = RidgeRegressionLeastSquares(l2_penalty=1.0, scale=True).fit(self.train)
        np.testing.assert_allclose(gd.theta_zero, ls.theta_zero, rtol=1e-3)
        np.testing.assert_allclose(gd.theta, ls.theta, rtol=1e-2, atol=1e-1)

    def test_predict(self):
        model = RidgeRegression(l2_penalty=1.0, alpha=0.01, max_iter=2000, patience=5, scale=True)
        model.fit(self.train)

        predictions = model.predict(self.test)
        self.assertEqual(predictions.shape[0], self.test.X.shape[0])

    def test_score_and_cost(self):
        model = RidgeRegression(l2_penalty=1.0, alpha=0.01, max_iter=2000, patience=5, scale=True)
        model.fit(self.train)

        score = model.score(self.test)
        cost = model.cost(self.test)

        print(f"\nTraining score (MSE): {model.score(self.train):.4f}")
        print(f"Test score  (MSE): {score:.4f}")
        print(f"Training cost: {model.cost(self.train):.4f}")
        print(f"Test cost : {cost:.4f}")

        self.assertTrue(np.isfinite(score))
        self.assertGreaterEqual(score, 0)
        self.assertTrue(np.isfinite(cost))


if __name__ == "__main__":
    unittest.main()