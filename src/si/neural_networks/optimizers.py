from abc import ABCMeta, abstractmethod

import numpy as np


class Optimizer(metaclass=ABCMeta):
    """
    Base class for the optimizers.
    An optimizer updates the parameters (weights or biases) of a layer
    using the gradient of the loss with respect to those parameters.
    """

    def __init__(self, learning_rate: float):
        """
        Parameters
        ----------
        learning_rate: float
            The learning rate (step size) of the updates
        """
        self.learning_rate = learning_rate

    @abstractmethod
    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Returns the updated parameters.

        Parameters
        ----------
        w: np.ndarray
            The current parameters
        grad_loss_w: np.ndarray
            The gradient of the loss with respect to the parameters (dE/dW)

        Returns
        -------
        np.ndarray
            The updated parameters
        """
        raise NotImplementedError


class SGD(Optimizer):
    """
    Stochastic Gradient Descent with momentum.

        V_t = momentum * V_{t-1} + (1 - momentum) * dE/dW
        W_t = W_{t-1} - learning_rate * V_t
    """

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        """
        Parameters
        ----------
        learning_rate: float
            The learning rate (step size) of the updates
        momentum: float
            Fraction of the previous update that is kept (0 = plain SGD)
        """
        super().__init__(learning_rate)
        self.momentum = momentum
        self.retained_gradient = None

    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        # first update: no previous gradient yet
        if self.retained_gradient is None:
            self.retained_gradient = np.zeros(np.shape(w))

        # moving average of the gradients (momentum)
        self.retained_gradient = self.momentum * self.retained_gradient + (1 - self.momentum) * grad_loss_w

        # step in the opposite direction of the gradient
        return w - self.learning_rate * self.retained_gradient


if __name__ == '__main__':
    # minimise f(w) = w^2 (gradient 2w): w should go towards 0
    sgd = SGD(learning_rate=0.1, momentum=0.9)
    w = np.array([5.0])
    for _ in range(200):
        w = sgd.update(w, 2 * w)
    print("w after 200 updates:", w)