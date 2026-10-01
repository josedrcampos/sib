import numpy as np

from si.data.dataset import Dataset


def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42):
    """
    Splits a dataset into train and test sets (random split, reproducible via random_state).

    Parameters
    ----------
    dataset: Dataset
        The dataset to split
    test_size: float
        Proportion of samples to use for the test set
    random_state: int
        Seed for the random number generator

    Returns
    -------
    train, test: tuple[Dataset, Dataset]
    """
    rng = np.random.default_rng(random_state)
    n_samples = dataset.X.shape[0]
    n_test = int(n_samples * test_size)

    permutation = rng.permutation(n_samples)
    test_idx = permutation[:n_test]
    train_idx = permutation[n_test:]

    train = Dataset(dataset.X[train_idx],
                    dataset.y[train_idx] if dataset.y is not None else None,
                    dataset.features,
                    dataset.label)
    test = Dataset(dataset.X[test_idx],
                   dataset.y[test_idx] if dataset.y is not None else None,
                   dataset.features,
                   dataset.label)
    return train, test
