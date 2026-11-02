from abc import ABCMeta, abstractmethod
import numpy as np
import copy

class Layer(metaclass=ABCMeta):

    @abstractmethod
    def forward_propagation(self, input, training):
        raise NotImplementedError

    @abstractmethod
    def backward_propagation(self, output_error):
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

    def initialize(self, optimizer):
        n_features = self.input_shape()[0]
        # pesos (n_features, n_units) uniformes em [-0.5, 0.5[
        self.weights = np.random.rand(n_features, self.n_units) - 0.5
        # bias (1, n_units) a zero
        self.biases = np.zeros((1, self.n_units))
        # uma cópia do optimizer para os pesos e outra para os bias
        self.w_opt = copy.deepcopy(optimizer)
        self.b_opt = copy.deepcopy(optimizer)
        return self

    def parameters(self):
        return self.weights.size + self.biases.size

    def forward_propagation(self, input, training=False):
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output

    def backward_propagation(self, output_error):
        input_error = np.dot(output_error, self.weights.T)            # dE/dX = dE/dY . W^T
        weights_error = np.dot(self.input.T, output_error)            # dE/dW = X^T . dE/dY
        bias_error = np.sum(output_error, axis=0, keepdims=True)      # dE/dB = dE/dY
        self.weights = self.w_opt.update(self.weights, weights_error)
        self.biases = self.b_opt.update(self.biases, bias_error)
        return input_error
    
    def output_shape(self):
        return (self.n_units,)
    
class Dropout(Layer):

    def __init__(self, probability: float):
        if not 0 <= probability < 1:
            raise ValueError("probability must be in [0, 1[")
        self.probability = probability
        self._input_shape = None
        self.mask = None
        self.input = None
        self.output = None

    def forward_propagation(self, input: np.ndarray, training: bool) -> np.ndarray:
        self.input = input

        if training:
            scaling_factor = 1 / (1 - self.probability)
            self.mask = np.random.binomial(1, 1 - self.probability, size=input.shape)
            self.output = input * self.mask * scaling_factor
            return self.output

        return input

    def backward_propagation(self, output_error: np.ndarray) -> np.ndarray:
        return output_error * self.mask

    def output_shape(self) -> tuple:
        return self._input_shape

    def parameters(self) -> int:
        return 0


if __name__ == "__main__":
    class NoUpdate:  # optimizer "falso": não mexe nos pesos
        def update(self, w, grad):
            return w

    layer = DenseLayer(4, (3,)).initialize(NoUpdate())
    out = layer.forward_propagation(np.random.rand(10, 3))
    assert out.shape == (10, 4)
    assert layer.parameters() == 3 * 4 + 4   # 16
    assert layer.output_shape() == (4,)

    # backward devolve o erro com a shape do input
    error = layer.backward_propagation(np.ones((10, 4)))
    assert error.shape == (10, 3)
    print("Todos os testes passaram.")



