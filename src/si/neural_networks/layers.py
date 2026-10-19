from abc import ABCMeta, abstractmethod
import numpy as np


class Layer(metaclass=ABCMeta):

    @abstractmethod
    def forward_propagation(self, input, training):
        raise NotImplementedError

    @abstractmethod
    def output_shape(self):
        raise NotImplementedError

    @abstractmethod
    def parameters(self):
        raise NotImplementedError

    def set_input_shape(self, input_shape):
        self._input_shape = input_shape

    def input_shape(self):
        return self._input_shape

    def layer_name(self):
        return self.__class__.__name__


class DenseLayer(Layer):

    def __init__(self, n_units, input_shape=None):
        self.n_units = n_units
        self._input_shape = input_shape
        self.input = None
        self.output = None
        self.weights = None
        self.biases = None

    def initialize(self):
        n_features = self.input_shape()[0]
        # pesos (n_features, n_units) uniformes em [-0.5, 0.5[
        self.weights = np.random.rand(n_features, self.n_units) - 0.5
        # bias (1, n_units) a zero
        self.biases = np.zeros((1, self.n_units))
        return self

    def parameters(self):
        return self.weights.size + self.biases.size

    def forward_propagation(self, input, training=False):
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output

    def output_shape(self):
        return (self.n_units,)


if __name__ == "__main__":
    layer = DenseLayer(4, (3,)).initialize()
    out = layer.forward_propagation(np.random.rand(10, 3))
    assert out.shape == (10, 4)
    assert layer.parameters() == 3 * 4 + 4   # 16
    assert layer.output_shape() == (4,)
    print("Todos os testes passaram.")



