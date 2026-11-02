import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.statistics.sigmoid_function import sigmoid_function


class LogisticRegression(Model):
    """
    The LogisticRegression is a logistic model using the L2 regularization.
    This model solves the logistic regression problem using an adapted Gradient Descent technique.
    It is a binary classifier: y must be 0 or 1.
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

    def _fit(self, dataset: Dataset) -> 'LogisticRegression':
        """
        Estimates theta, theta_zero, mean, std and cost_history with gradient descent.
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

        # initialize the parameters at zero
        self.theta = np.zeros(n)
        self.theta_zero = 0.0
        self.cost_history = {}

        early_stopping = 0
        for i in range(self.max_iter):
            # 2. estimate y with the sigmoid: probability of class 1
            y_pred = sigmoid_function(X.dot(self.theta) + self.theta_zero)

            # 3. gradient multiplied by the learning rate
            #    (same form as in linear regression, but h(x) is now the sigmoid)
            gradient = (self.alpha / m) * X.T.dot(y_pred - y)

            # 4. L2 regularization term
            penalization_term = self.theta * (1 - self.alpha * (self.l2_penalty / m))

            # 5. update theta
            self.theta = penalization_term - gradient

            # 6. update theta_zero (not regularized)
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
        J(theta) = -1/m * sum( y*log(h) + (1-y)*log(1-h) ) + lambda/(2m) * sum(theta_j^2)
        Receives X already scaled.
        """
        m = X_scaled.shape[0]
        p = sigmoid_function(X_scaled.dot(self.theta) + self.theta_zero)
        # clip to avoid log(0) when the sigmoid saturates at exactly 0 or 1
        p = np.clip(p, 1e-15, 1 - 1e-15)
        cross_entropy = -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)) / m
        regularization = (self.l2_penalty / (2 * m)) * np.sum(self.theta ** 2)
        return float(cross_entropy + regularization)

    def cost(self, dataset: Dataset) -> float:
        """
        Cost function J between the real and predicted y values of a dataset.
        """
        X = self._scale(np.asarray(dataset.X, dtype=float))
        y = np.asarray(dataset.y, dtype=float).reshape(-1)
        return self._cost(X, y)

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts the class (0 or 1) of every sample.
        """
        X = self._scale(np.asarray(dataset.X, dtype=float))

        # 1. probabilities with the sigmoid
        probabilities = sigmoid_function(X.dot(self.theta) + self.theta_zero)

        # 2. probability >= 0.5 -> 1, otherwise 0
        return np.where(probabilities >= 0.5, 1, 0)

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Accuracy between the real and predicted classes.
        """
        return accuracy(dataset.y, predictions)


if __name__ == '__main__':
    import os
    from datasets import DATASETS_PATH
    from si.io.data_file import read_data_file
    from si.model_selection.split import stratified_train_test_split

    # breast-bin.csv has no header row, so it is read with read_data_file
    breast = read_data_file(os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv'),
                            sep=',', label=True)
    train, test = stratified_train_test_split(breast, test_size=0.2, random_state=42)

    model = LogisticRegression(l2_penalty=1.0, alpha=0.001, max_iter=2000, patience=5, scale=True)
    model.fit(train)
    print(f'Iterations: {len(model.cost_history)}')
    print(f'Test accuracy: {model.score(test):.4f}')
    print(f'Test cost: {model.cost(test):.4f}')
