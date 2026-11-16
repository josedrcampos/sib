import itertools
from typing import Callable, Dict

import numpy as np

from si.data.dataset import Dataset
from si.model_selection._search_utils import k_fold_indices, cross_validate_configuration


def grid_search_cv(model, dataset: Dataset, hyperparameter_grid: Dict[str, list],
                   scoring: Callable = None, cv: int = 5, seed: int = None) -> Dict:
    """
    Exhaustive search: cross-validates every combination of the hyperparameter values.

    Parameters
    ----------
    model: Model or Callable
        The model to tune, or a builder function that returns a new model
    dataset: Dataset
        The training dataset (never the test set)
    hyperparameter_grid: dict
        {name: list of values to try}
    scoring: Callable
        scoring(y_true, y_pred); if None, uses model.score
    cv: int
        Number of folds
    seed: int
        Seed used to create the folds

    Returns
    -------
    results: dict
        hyperparameters: list with every configuration tested
        scores: mean validation score of each configuration
        best_hyperparameters: the configuration with the highest mean score
        best_score: its mean score
    """
    # 1. the k folds, created once (the same folds for every configuration)
    folds = k_fold_indices(dataset.shape()[0], cv=cv, seed=seed)

    names = list(hyperparameter_grid.keys())
    results = {'hyperparameters': [], 'scores': []}

    # 2. every combination of the hyperparameter values
    for values in itertools.product(*hyperparameter_grid.values()):
        configuration = dict(zip(names, values))

        # 3. cross-validate this configuration (a fresh model per fold)
        score = cross_validate_configuration(model, dataset, configuration, folds, scoring)

        results['hyperparameters'].append(configuration)
        results['scores'].append(score)

    # 4. the configuration with the highest mean score
    best = int(np.argmax(results['scores']))
    results['best_hyperparameters'] = results['hyperparameters'][best]
    results['best_score'] = results['scores'][best]

    # 5. results dictionary
    return results


if __name__ == '__main__':
    import os

    from si.io.csv_file import read_csv
    from si.models.knn_classifier import KNNClassifier

    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    iris = read_csv(os.path.join(ROOT, 'datasets', 'iris', 'iris.csv'), features=True, label=True)

    results = grid_search_cv(KNNClassifier(), iris, {'k': [1, 3, 5, 7, 15]}, cv=5, seed=1)
    for h, s in zip(results['hyperparameters'], results['scores']):
        print(h, round(s, 3))
    print("best:", results['best_hyperparameters'], round(results['best_score'], 3))