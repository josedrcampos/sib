import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.mse import mse


class RidgeRegression(Model):
    """
    Ridge Regression (linear regression with L2 regularization),
    solved with the analytical method (regularized normal equation).
    """

    def __init__(self, l2_penalty: float = 1.0, alpha: float = 0.01,
                 max_iter: int = 1000, patience: int = 10, scale: bool = True, **kwargs):
        """
        Parameters
        ----------
        l2_penalty: float
            L2 regularization parameter
        alpha: float
            Learning rate (not used in the analytical method)
        max_iter: int
            Maximum number of iterations (not used in the analytical method)
        patience: int
            Maximum number of iterations without improvement (not used in the analytical method)
        scale: bool
            Whether to scale the data or not
        """
        super().__init__(**kwargs)
        # Parameters
        self.l2_penalty = l2_penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.patience = patience
        self.scale = scale

        # Estimated parameters
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None
        self.cost_history = {}

    def cost(self, dataset: Dataset) -> float:
        """
        Computes the cost function J with L2 regularization
        (the intercept is not penalized).

        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the cost on.

        Returns
        -------
        cost: float
        """
        y_pred = self._predict(dataset)
        y_true = np.asarray(dataset.y, dtype=float).reshape(-1)
        return self._cost_from_predictions(y_true, y_pred)

    def _cost_from_predictions(self, y_true, y_pred) -> float:
        """
        Computes J = 1/(2m) * [sum((y - y_pred)^2) + lambda * sum(theta^2)]
        from the real and predicted values.
        """
        y_true = np.asarray(y_true, dtype=float).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
        m = len(y_true)
        mse_cost = (1 / (2 * m)) * np.sum((y_true - y_pred) ** 2)
        l2_cost = (self.l2_penalty / (2 * m)) * np.sum(self.theta ** 2)
        return mse_cost + l2_cost

    def _fit(self, dataset: Dataset) -> "RidgeRegression":
        """
        Estimates theta, theta_zero, mean, std and cost_history using the
        closed-form solution: theta = (X^T X + lambda * I')^-1 X^T y,
        where I' is the identity matrix with the intercept entry set to zero.

        Parameters
        ----------
        dataset: Dataset
            The training dataset.

        Returns
        -------
        self: RidgeRegression
        """
        x = np.asarray(dataset.X, dtype=float)
        y = np.asarray(dataset.y, dtype=float).reshape(-1, 1)
        m, n = x.shape

        # 1. Scale the data (if required)
        if self.scale:
            self.mean = np.mean(x, axis=0)
            self.std = np.std(x, axis=0)
            self.std[self.std == 0] = 1.0  # avoid division by zero
            x_scaled = (x - self.mean) / self.std
        else:
            self.mean = np.zeros(n)
            self.std = np.ones(n)
            x_scaled = x.copy()

        # 2. Design matrix with a column of ones (intercept at position 0)
        x_b = np.c_[np.ones(m), x_scaled]

        # 3. Penalty matrix: identity, but the intercept is not penalized
        penalty = self.l2_penalty * np.eye(n + 1)
        penalty[0, 0] = 0.0

        # 4. Analytical solution (solve is more stable than inverting the matrix)
        params = np.linalg.solve(x_b.T @ x_b + penalty, x_b.T @ y).reshape(-1)

        self.theta_zero = params[0]
        self.theta = params[1:]

        # 5. Cost (single "iteration")
        y_pred = x_scaled @ self.theta + self.theta_zero
        self.cost_history = {1: self._cost_from_predictions(y.reshape(-1), y_pred)}

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts y using theta and theta_zero (applying the same scaling used in training).

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict on.

        Returns
        -------
        predictions: np.ndarray
        """
        x = np.asarray(dataset.X, dtype=float)
        x_scaled = (x - self.mean) / self.std
        return x_scaled @ self.theta + self.theta_zero

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Computes the error (MSE) between the real and predicted y values.

        Parameters
        ----------
        dataset: Dataset
            The dataset with the real y values.
        predictions: np.ndarray
            The predicted y values.

        Returns
        -------
        score: float
        """
        return mse(dataset.y, predictions)
