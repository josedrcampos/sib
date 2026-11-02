import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.mse import mse


class RidgeRegressionLeastSquares(Model):
    """
    Ridge Regression (linear regression with L2 regularization) solved with the
    analytical method of least squares (regularized normal equation):

        thetas = (X^T X + lambda * I')^-1 X^T y

    where X has a first column of ones (intercept) and I' is the identity matrix
    with I'[0, 0] = 0, so that the intercept is not penalized.
    """

    def __init__(self, l2_penalty: float = 1.0, scale: bool = True, **kwargs):
        """
        Parameters
        ----------
        l2_penalty: float
            The L2 regularization parameter (lambda)
        scale: bool
            Whether to scale the dataset (z-score) or not
        """
        super().__init__(**kwargs)
        # parameters
        self.l2_penalty = l2_penalty
        self.scale = scale

        # estimated parameters
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None

    def _scale(self, X: np.ndarray) -> np.ndarray:
        """
        Applies the scaling estimated in _fit (mean and std of the TRAINING data).
        """
        if self.scale:
            return (X - self.mean) / self.std
        return X

    def _fit(self, dataset: Dataset) -> 'RidgeRegressionLeastSquares':
        """
        Estimates theta, theta_zero, mean and std with the closed-form solution.
        """
        X = np.asarray(dataset.X, dtype=float)
        y = np.asarray(dataset.y, dtype=float).reshape(-1)
        m, n = X.shape

        # 1. scale the data if required
        if self.scale:
            self.mean = np.nanmean(X, axis=0)
            self.std = np.nanstd(X, axis=0)
            self.std[self.std == 0] = 1.0
        X = self._scale(X)

        # 2. add the intercept term: a column of ones in the first position
        X = np.c_[np.ones(m), X]

        # 3. penalty matrix: lambda * identity
        penalty = self.l2_penalty * np.eye(n + 1)

        # 4. do not penalize the intercept (theta_zero)
        penalty[0, 0] = 0

        # 5. thetas = (X^T X + penalty)^-1 X^T y
        #    np.linalg.solve gives the same result as np.linalg.inv(A).dot(b),
        #    but solves the linear system directly (faster and numerically more stable)
        thetas = np.linalg.solve(X.T.dot(X) + penalty, X.T.dot(y))

        self.theta_zero = thetas[0]
        self.theta = thetas[1:]
        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts y using theta and theta_zero.
        """
        # 1. scale with the mean and std estimated in _fit
        X = self._scale(np.asarray(dataset.X, dtype=float))

        # 2. add the intercept term
        X = np.c_[np.ones(X.shape[0]), X]

        # 3. join theta_zero and theta in one vector and compute X * thetas
        thetas = np.r_[self.theta_zero, self.theta]
        return X.dot(thetas)

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        MSE between the real and predicted values of y.
        """
        return mse(dataset.y, predictions)


if __name__ == '__main__':
    import os
    from datasets import DATASETS_PATH
    from si.io.csv_file import read_csv
    from si.model_selection.split import train_test_split

    cpu = read_csv(os.path.join(DATASETS_PATH, 'cpu', 'cpu.csv'), features=True, label=True)
    train, test = train_test_split(cpu, test_size=0.2, random_state=42)

    model = RidgeRegressionLeastSquares(l2_penalty=1.0, scale=True)
    model.fit(train)
    print(f'theta_zero: {model.theta_zero:.4f}')
    print(f'Test score (MSE): {model.score(test):.4f}')
