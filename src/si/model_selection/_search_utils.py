import copy
from typing import Callable, Dict, List, Tuple

import numpy as np

from si.data.dataset import Dataset


def k_fold_indices(n_samples: int, cv: int = 5, seed: int = None) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Splits the sample indexes into k folds.
    Each fold is used once as validation set while the other k-1 form the training set.

    Parameters
    ----------
    n_samples: int
        Number of samples in the dataset
    cv: int
        Number of folds (k)
    seed: int
        Seed for the permutation (same seed -> same folds)

    Returns
    -------
    folds: list of (train_idx, val_idx)
    """
    if cv < 2 or cv > n_samples:
        raise ValueError("cv must be between 2 and the number of samples")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    # k parts of (almost) equal size
    parts = np.array_split(indices, cv)

    folds = []
    for i in range(cv):
        val_idx = parts[i]
        train_idx = np.concatenate([parts[j] for j in range(cv) if j != i])
        folds.append((train_idx, val_idx))
    return folds


def build_model(model, hyperparameters: Dict):
    """
    Returns a new, untrained model with the given hyperparameters.

    - If model is a builder function (or a class), it is called with the hyperparameters:
      model(**hyperparameters). Needed for the NeuralNetwork, whose optimizer is created in __init__.
    - If model is a model object, it is deep-copied and the hyperparameters are set as attributes.

    Parameters
    ----------
    model: Model or Callable
        The model to copy, or a function that builds a new model
    hyperparameters: dict
        {name: value}

    Returns
    -------
    A fresh model
    """
    if callable(model):
        return model(**hyperparameters)

    new_model = copy.deepcopy(model)
    for name, value in hyperparameters.items():
        if not hasattr(new_model, name):
            raise AttributeError(f"Model {type(new_model).__name__} has no hyperparameter '{name}'")
        setattr(new_model, name, value)
    return new_model


def _subset(dataset: Dataset, idx: np.ndarray) -> Dataset:
    """
    Returns a new Dataset with only the samples in idx.
    """
    return Dataset(dataset.X[idx], dataset.y[idx], features=dataset.features, label=dataset.label)


def cross_validate_configuration(model, dataset: Dataset, hyperparameters: Dict,
                                 folds: List[Tuple[np.ndarray, np.ndarray]],
                                 scoring: Callable = None) -> float:
    """
    Cross-validates one configuration of hyperparameters:
    for each fold, builds a fresh model, fits it on the training part and scores it on the validation part.

    Parameters
    ----------
    model: Model or Callable
        The model to tune, or a builder function
    dataset: Dataset
        The training dataset
    hyperparameters: dict
        The configuration to evaluate
    folds: list of (train_idx, val_idx)
        The folds (from k_fold_indices), the same for every configuration
    scoring: Callable
        scoring(y_true, y_pred); if None, uses model.score

    Returns
    -------
    float
        Mean of the k validation scores
    """
    scores = []
    for train_idx, val_idx in folds:
        train = _subset(dataset, train_idx)
        val = _subset(dataset, val_idx)

        # a fresh model for every fold: never reuse the weights of another fold
        fold_model = build_model(model, hyperparameters)
        fold_model.fit(train)

        if scoring is None:
            scores.append(fold_model.score(val))
        else:
            scores.append(scoring(val.y, fold_model.predict(val)))

    return float(np.mean(scores))


if __name__ == '__main__':
    import os

    from si.io.csv_file import read_csv
    from si.models.knn_classifier import KNNClassifier

    # repository root: src/si/model_selection -> 3 levels up
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    DATASETS_PATH = os.path.join(ROOT, 'datasets')

    folds = k_fold_indices(10, cv=3, seed=1)
    for train_idx, val_idx in folds:
        print("train:", train_idx, "val:", val_idx)

    iris = read_csv(os.path.join(DATASETS_PATH, 'iris', 'iris.csv'), features=True, label=True)
    folds = k_fold_indices(iris.shape()[0], cv=5, seed=1)
    for k in (1, 5, 15):
        score = cross_validate_configuration(KNNClassifier(), iris, {'k': k}, folds)
        print(f"k={k}: mean accuracy = {score:.3f}")