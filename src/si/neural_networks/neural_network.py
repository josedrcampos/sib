from typing import Callable, Iterator, Tuple

import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.mse import mse
from si.neural_networks.layers import Layer
from si.neural_networks.losses import LossFunction, MeanSquaredError, BinaryCrossEntropy, CategoricalCrossEntropy
from si.neural_networks.optimizers import Optimizer, SGD


class NeuralNetwork(Model):
    """
    Feed-forward neural network: a sequence of layers trained with mini-batch gradient descent.
    """

    def __init__(self, epochs: int = 100, batch_size: int = 128, optimizer: type = SGD,
                 learning_rate: float = 0.01, verbose: bool = False, loss: type = MeanSquaredError,
                 metric: Callable = mse, **kwargs):
        """
        Parameters
        ----------
        epochs: int
            Number of passes over the training data
        batch_size: int
            Number of samples in each mini-batch
        optimizer: type
            Optimizer class (e.g. SGD); it is created here with learning_rate and the extra kwargs (e.g. momentum)
        learning_rate: float
            Learning rate of the optimizer
        verbose: bool
            Whether to print the loss and metric of each epoch
        loss: type
            Loss function class (e.g. BinaryCrossEntropy)
        metric: Callable
            Metric function metric(y_true, y_pred) used to evaluate the network
        **kwargs
            Extra arguments for the optimizer (e.g. momentum=0.9)
        """
        super().__init__()
        # parameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer: Optimizer = optimizer(learning_rate=learning_rate, **kwargs)
        self.verbose = verbose
        self.loss: LossFunction = loss()
        self.metric = metric

        # attributes
        self.layers = []
        self.history = {}

    def add(self, layer: Layer) -> 'NeuralNetwork':
        """
        Adds a layer to the network. The input shape of the layer is the output shape of the previous one,
        and layers with parameters (DenseLayer) get their own copy of the optimizer.
        """
        if self.layers:
            layer.set_input_shape(input_shape=self.layers[-1].output_shape())
        if hasattr(layer, 'initialize'):
            layer.initialize(self.optimizer)
        self.layers.append(layer)
        return self

    def _get_mini_batches(self, X: np.ndarray, y: np.ndarray = None,
                          shuffle: bool = True) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Yields the data in mini-batches of batch_size samples (the last one can be smaller).
        """
        n_samples = X.shape[0]
        indices = np.random.permutation(n_samples) if shuffle else np.arange(n_samples)
        for start in range(0, n_samples, self.batch_size):
            batch = indices[start:start + self.batch_size]
            yield X[batch], (y[batch] if y is not None else None)

    def _forward_propagation(self, X: np.ndarray, training: bool) -> np.ndarray:
        output = X
        for layer in self.layers:
            output = layer.forward_propagation(output, training)
        return output

    def _backward_propagation(self, output_error: np.ndarray) -> np.ndarray:
        error = output_error
        for layer in reversed(self.layers):
            error = layer.backward_propagation(error)
        return error

    def _prepare_y(self, y: np.ndarray, n_outputs: int) -> np.ndarray:
        """
        Puts y in the shape of the network output: (n_samples, 1) or one-hot (n_samples, n_classes).
        """
        y = np.asarray(y)
        if isinstance(self.loss, CategoricalCrossEntropy) and y.ndim == 1:
            one_hot = np.zeros((y.shape[0], n_outputs))
            one_hot[np.arange(y.shape[0]), y.astype(int)] = 1
            return one_hot
        return y.reshape(-1, 1).astype(float) if y.ndim == 1 else y.astype(float)

    def _to_predictions(self, output: np.ndarray) -> np.ndarray:
        """
        Converts the raw output of the network into predictions:
        classes for classification (threshold 0.5 or argmax), values for regression.
        """
        if isinstance(self.loss, BinaryCrossEntropy):
            return (output >= 0.5).astype(int).ravel()
        if isinstance(self.loss, CategoricalCrossEntropy):
            return np.argmax(output, axis=1)
        return output.ravel() if output.shape[1] == 1 else output

    def _fit(self, dataset: Dataset) -> 'NeuralNetwork':
        """
        Trains the network: for each epoch, forward and backward propagation over every mini-batch.
        """
        X = dataset.X
        y = self._prepare_y(dataset.y, self.layers[-1].output_shape()[0])
        self.history = {}

        for epoch in range(1, self.epochs + 1):
            # store the outputs of the epoch to compute the loss and metric over all the samples
            output_x_ = []
            y_ = []
            for X_batch, y_batch in self._get_mini_batches(X, y):
                # forward propagation
                output = self._forward_propagation(X_batch, training=True)
                # backward propagation, starting from the derivative of the loss
                error = self.loss.derivative(y_batch, output)
                self._backward_propagation(error)

                output_x_.append(output)
                y_.append(y_batch)

            output_x_all = np.concatenate(output_x_)
            y_all = np.concatenate(y_)

            # loss and metric of the epoch
            loss = self.loss.loss(y_all, output_x_all)
            if self.metric is not None:
                metric = self.metric(self._to_predictions(y_all) if y_all.shape[1] > 1 else y_all.ravel(),
                                     self._to_predictions(output_x_all))
                metric_s = f"{self.metric.__name__}: {metric:.4f}"
            else:
                metric = None
                metric_s = "NA"

            self.history[epoch] = {'loss': loss, 'metric': metric}

            if self.verbose:
                print(f"Epoch {epoch}/{self.epochs} - loss: {loss:.4f} - {metric_s}")

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts the output for the samples of the dataset (forward propagation only, training=False).
        """
        output = self._forward_propagation(dataset.X, training=False)
        return self._to_predictions(output)

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Computes the metric between the real values and the predictions.
        """
        if self.metric is None:
            raise ValueError("No metric specified for the neural network.")
        return self.metric(dataset.y, predictions)


if __name__ == '__main__':
    from si.metrics.accuracy import accuracy
    from si.model_selection.split import train_test_split
    from si.neural_networks.activation import ReLUActivation, SigmoidActivation
    from si.neural_networks.layers import DenseLayer

    np.random.seed(0)
    # binary problem: is the point inside the circle?
    rng = np.random.default_rng(42)
    X = rng.uniform(-2, 2, size=(400, 2))
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 < 1.5).astype(float)
    train, test = train_test_split(Dataset(X, y), test_size=0.2)

    net = NeuralNetwork(epochs=200, batch_size=16, optimizer=SGD, learning_rate=0.1, momentum=0.9,
                        loss=BinaryCrossEntropy, metric=accuracy, verbose=False)
    net.add(DenseLayer(16, (2,)))
    net.add(ReLUActivation())
    net.add(DenseLayer(1))
    net.add(SigmoidActivation())

    net.fit(train)
    print("final train loss:", round(net.history[net.epochs]['loss'], 4))
    print("test accuracy:", net.score(test))