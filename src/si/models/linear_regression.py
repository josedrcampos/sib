import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.mse import mse


class RidgeRegression(Model):
    """
    The RidgeRegression is a linear model using the L2 regularization.
    This model solves the linear regression problem using an adapted Gradient Descent technique.
    """

    def __init__(self, l2_penalty: float = 1.0, alpha: float = 0.001,
                 max_iter: int = 1000, patience: int = 5, scale: bool = True, **kwargs):
        """
        Parameters
        ----------
        l2_penalty: float
            The L2 regularization parameter (lambda)
        alpha: float
            The learning rate (size of each gradient descent step)
        max_iter: int
            The maximum number of iterations
        patience: int
            Maximum number of consecutive iterations without improvement of the cost
        scale: bool
            Whether to scale the dataset (z-score) or not
        """
        super().__init__(**kwargs)
        # parameters
        self.l2_penalty = l2_penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.patience = patience
        self.scale = scale

        # estimated parameters
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None
        self.cost_history = {}

    def _scale(self, X: np.ndarray) -> np.ndarray:
        """
        Applies the scaling estimated in _fit (mean and std of the TRAINING data).
        """
        if self.scale:
            return (X - self.mean) / self.std
        return X

    def _fit(self, dataset: Dataset) -> 'RidgeRegression':
        """
        Estimates theta, theta_zero, mean, std and cost_history with gradient descent.
        """
        X = np.asarray(dataset.X, dtype=float)
        y = np.asarray(dataset.y, dtype=float).reshape(-1)
        m, n = X.shape

        # 1. scale the data if required: x' = (x - mean) / std
        if self.scale:
            self.mean = np.nanmean(X, axis=0)
            self.std = np.nanstd(X, axis=0)
            self.std[self.std == 0] = 1.0      # constant features: avoid division by zero
        X = self._scale(X)

        # initialize the parameters at zero
        self.theta = np.zeros(n)
        self.theta_zero = 0.0
        self.cost_history = {}

        early_stopping = 0
        for i in range(self.max_iter):
            # 2. predict y: h(x) = theta^T x + theta_zero
            y_pred = X.dot(self.theta) + self.theta_zero

            # 3. gradient of the error, already multiplied by the learning rate
            #    alpha * (1/m) * sum((h(x) - y) * x_j)
            gradient = (self.alpha / m) * X.T.dot(y_pred - y)

            # 4. L2 term: theta_j * (1 - alpha * lambda / m)  ("weight decay")
            penalization_term = self.theta * (1 - self.alpha * (self.l2_penalty / m))

            # 5. update theta
            self.theta = penalization_term - gradient

            # 6. update theta_zero (the intercept is NOT regularized; x_0 = 1)
            self.theta_zero = self.theta_zero - (self.alpha / m) * np.sum(y_pred - y)

            # 7. cost function with the new parameters
            self.cost_history[i] = self._cost(X, y)

            # 8. stop at max_iter or when the cost stops improving for `patience` iterations
            if i > 0 and self.cost_history[i] >= self.cost_history[i - 1]:
                early_stopping += 1
            else:
                early_stopping = 0
            if early_stopping >= self.patience:
                break

        return self

    def _cost(self, X_scaled: np.ndarray, y: np.ndarray) -> float:
        """
        J(theta) = 1/(2m) * [ sum((h(x) - y)^2) + lambda * sum(theta_j^2) ]
        Receives X already scaled.
        """
        m = X_scaled.shape[0]
        y_pred = X_scaled.dot(self.theta) + self.theta_zero
        return float((np.sum((y_pred - y) ** 2) + self.l2_penalty * np.sum(self.theta ** 2)) / (2 * m))

    def cost(self, dataset: Dataset) -> float:
        """
        Cost function J between the real and predicted y values of a dataset.
        """
        X = self._scale(np.asarray(dataset.X, dtype=float))
        y = np.asarray(dataset.y, dtype=float).reshape(-1)
        return self._cost(X, y)

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts y using theta and theta_zero (applying the same scaling as in training).
        """
        X = self._scale(np.asarray(dataset.X, dtype=float))
        return X.dot(self.theta) + self.theta_zero

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

    model = RidgeRegression(l2_penalty=1.0, alpha=0.01, max_iter=2000, patience=5, scale=True)
    model.fit(train)
    print(f'Iterations: {len(model.cost_history)}')
    print(f'Test score (MSE): {model.score(test):.4f}')
    print(f'Test cost: {model.cost(test):.4f}')
