from typing import Callable, Dict

import numpy as np

from si.data.dataset import Dataset
from si.model_selection._search_utils import k_fold_indices, cross_validate_configuration


def log_uniform(low: float, high: float) -> Callable:
    """
    Returns a sampler for values spread evenly over orders of magnitude:
    10 ** uniform(log10(low), log10(high)).

    Parameters
    ----------
    low: float
        Smallest value (> 0)
    high: float
        Largest value

    Returns
    -------
    sampler: Callable
        sampler(rng) -> one value between low and high
    """
    if low <= 0 or high <= low:
        raise ValueError("log_uniform needs 0 < low < high")

    def sampler(rng: np.random.Generator) -> float:
        return float(10 ** rng.uniform(np.log10(low), np.log10(high)))

    return sampler


def randomized_search_cv(model, dataset: Dataset, hyperparameter_distributions: Dict,
                         scoring: Callable = None, cv: int = 5, n_iter: int = 10,
                         seed: int = None) -> Dict:
    """
    Random search: cross-validates n_iter random configurations of the hyperparameters.

    Parameters
    ----------
    model: Model or Callable
        The model to tune, or a builder function that returns a new model
    dataset: Dataset
        The training dataset (never the test set)
    hyperparameter_distributions: dict
        {name: list of values (uniform choice) or sampler function (called with the generator)}
    scoring: Callable
        scoring(y_true, y_pred); if None, uses model.score
    cv: int
        Number of folds
    n_iter: int
        Number of random configurations to evaluate
    seed: int
        Seed for the folds and for the random generator

    Returns
    -------
    results: dict (same as grid_search_cv)
        hyperparameters, scores, best_hyperparameters, best_score
    """
    # the k folds, created once, and the random generator
    folds = k_fold_indices(dataset.shape()[0], cv=cv, seed=seed)
    rng = np.random.default_rng(seed)

    results = {'hyperparameters': [], 'scores': []}

    for _ in range(n_iter):
        # sample one value per hyperparameter
        configuration = {}
        for name, distribution in hyperparameter_distributions.items():
            if callable(distribution):
                configuration[name] = distribution(rng)
            else:
                value = distribution[rng.integers(len(distribution))]
                # numpy integers -> python int (e.g. for batch_size, epochs)
                configuration[name] = value.item() if isinstance(value, np.generic) else value

        # cross-validate this configuration (a fresh model per fold)
        score = cross_validate_configuration(model, dataset, configuration, folds, scoring)

        results['hyperparameters'].append(configuration)
        results['scores'].append(score)

    best = int(np.argmax(results['scores']))
    results['best_hyperparameters'] = results['hyperparameters'][best]
    results['best_score'] = results['scores'][best]
    return results


if __name__ == '__main__':
    import os

    from si.io.csv_file import read_csv
    from si.models.knn_classifier import KNNClassifier

    # log_uniform: the samples cover every order of magnitude
    sampler = log_uniform(1e-4, 1e-1)
    rng = np.random.default_rng(0)
    samples = np.array([sampler(rng) for _ in range(1000)])
    for a, b in [(1e-4, 1e-3), (1e-3, 1e-2), (1e-2, 1e-1)]:
        print(f"[{a:g}, {b:g}[: {np.mean((samples >= a) & (samples < b)):.0%}")

    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    iris = read_csv(os.path.join(ROOT, 'datasets', 'iris', 'iris.csv'), features=True, label=True)

    results = randomized_search_cv(KNNClassifier(), iris, {'k': list(range(1, 31))}, cv=5, n_iter=5, seed=1)
    for h, s in zip(results['hyperparameters'], results['scores']):
        print(h, round(s, 3))
    print("best:", results['best_hyperparameters'], round(results['best_score'], 3))