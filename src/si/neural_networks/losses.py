from abc import ABCMeta, abstractmethod

import numpy as np


class LossFunction(metaclass=ABCMeta):
    """
    Base class for the loss functions.
    loss() measures the error of the predictions; derivative() gives dE/dY_pred,
    which is the error that starts the backward propagation.
    """

    @abstractmethod
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        raise NotImplementedError

    @abstractmethod
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class MeanSquaredError(LossFunction):
    """
    Mean squared error, for regression.

        L = (1/n) * sum((y_true - y_pred)^2)
        dL/dy_pred = (2/n) * (y_pred - y_true)
    """

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean((y_true - y_pred) ** 2))

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return 2 * (y_pred - y_true) / y_true.size


class BinaryCrossEntropy(LossFunction):
    """
    Binary cross entropy, for binary classification (output of a sigmoid).

        L = -(1/n) * sum(y_true * log(p) + (1 - y_true) * log(1 - p))
        dL/dp = (1/n) * (-y_true / p + (1 - y_true) / (1 - p))
    """

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # avoid log(0)
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        # avoid division by 0
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return (-(y_true / p) + (1 - y_true) / (1 - p)) / y_true.size


class CategoricalCrossEntropy(LossFunction):
    """
    Categorical cross entropy, for multi-class classification
    (y_true one-hot encoded, y_pred = output of a softmax).

        L = -(1/n) * sum(y_true * log(p))
        dL/dp = -(1/n) * y_true / p
    """

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return float(-np.sum(y_true * np.log(p)) / y_true.shape[0])

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -(y_true / p) / y_true.shape[0]


if __name__ == '__main__':
    y_true = np.array([[1.0], [0.0], [1.0], [0.0]])
    y_pred = np.array([[0.9], [0.2], [0.6], [0.1]])

    for loss_func in (MeanSquaredError(), BinaryCrossEntropy()):
        # compare the analytical derivative with a numerical one (finite differences)
        eps = 1e-6
        numerical = np.zeros_like(y_pred)
        for i in range(y_pred.shape[0]):
            plus, minus = y_pred.copy(), y_pred.copy()
            plus[i] += eps
            minus[i] -= eps
            numerical[i] = (loss_func.loss(y_true, plus) - loss_func.loss(y_true, minus)) / (2 * eps)

        analytical = loss_func.derivative(y_true, y_pred)
        print(loss_func.__class__.__name__,
            "loss:", round(loss_func.loss(y_true, y_pred), 4),
            "| derivative OK:", np.allclose(analytical, numerical))