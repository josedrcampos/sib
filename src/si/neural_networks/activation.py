    from abc import abstractmethod

import numpy as np

from si.neural_networks.layers import Layer
from si.statistics.sigmoid_function import sigmoid_function


class ActivationLayer(Layer):

    def __init__(self):
        self._input_shape = None
        self.input = None
        self.output = None

    def forward_propagation(self, input, training=False):
        self.input = input
        self.output = self.activation_function(self.input)
        return self.output

    def backward_propagation(self, output_error):
        # dE/dX = f'(X) * dE/dY  (elemento a elemento, não np.dot)
        return self.derivative(self.input) * output_error

    @abstractmethod
    def activation_function(self, input):
        raise NotImplementedError

    @abstractmethod
    def derivative(self, input):
        raise NotImplementedError

    def output_shape(self):
        return self._input_shape

    def parameters(self):
        return 0


class ReLUActivation(ActivationLayer):

    def activation_function(self, input):
        return np.maximum(0, input)

    def derivative(self, input):
        return np.where(input > 0, 1, 0)


class SigmoidActivation(ActivationLayer):

    def activation_function(self, input):
        return sigmoid_function(input)

    def derivative(self, input):
        f_x = sigmoid_function(input)
        return f_x * (1 - f_x)
    
class TanhActivation(ActivationLayer):

    def activation_function(self, input: np.ndarray) -> np.ndarray:
        return np.tanh(input)

    def derivative(self, input: np.ndarray) -> np.ndarray:
        # f'(x) = 1 - f(x)^2
        return 1 - np.tanh(input) ** 2


class SoftmaxActivation(ActivationLayer):

    def activation_function(self, input: np.ndarray) -> np.ndarray:
        # versão estável: subtrair o máximo de cada linha antes da exponencial
        exp_values = np.exp(input - np.max(input, axis=-1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=-1, keepdims=True)

    def derivative(self, input: np.ndarray) -> np.ndarray:
        # f'(x) = f(x) * (1 - f(x))
        f_x = self.activation_function(input)
        return f_x * (1 - f_x)