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


def stratified_train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42):
    """
    Splits a dataset into train and test sets keeping the class proportions
    of the original dataset in both sets.

    Parameters
    ----------
    dataset: Dataset
        The (labeled) dataset to split
    test_size: float
        Proportion of samples of each class to use for the test set
    random_state: int
        Seed for the random number generator

    Returns
    -------
    train, test: tuple[Dataset, Dataset]
    """
    if not dataset.has_label():
        raise ValueError("Stratified split needs a dataset with a label (y)")

    rng = np.random.default_rng(random_state)

    # 1. unique class labels and how many samples each one has
    labels, counts = np.unique(dataset.y, return_counts=True)

    # 2. empty lists for the train and test indexes
    train_idx, test_idx = [], []

    # 3. loop through the classes
    for label, count in zip(labels, counts):
        # 4. number of test samples for this class
        n_test = int(count * test_size)

        # 5. shuffle the indexes of this class and take the first n_test for the test set
        class_idx = np.where(dataset.y == label)[0]
        class_idx = rng.permutation(class_idx)
        test_idx.extend(class_idx[:n_test])

        # 6. the remaining indexes go to the train set
        train_idx.extend(class_idx[n_test:])

    train_idx = np.array(train_idx, dtype=int)
    test_idx = np.array(test_idx, dtype=int)

    # 7. build the train and test datasets
    train = Dataset(dataset.X[train_idx], dataset.y[train_idx], dataset.features, dataset.label)
    test = Dataset(dataset.X[test_idx], dataset.y[test_idx], dataset.features, dataset.label)
    return train, test
