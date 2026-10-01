import numpy as np

from si.data.dataset import Dataset


def read_data_file(filename: str, sep: str = ',', label: bool = False) -> Dataset:
    """
    Reads a data file without a header using numpy.genfromtxt.
    label: True if the last column is the dependent variable (y).
    """
    data = np.genfromtxt(filename, delimiter=sep)

    if label:
        x = data[:, :-1]
        y = data[:, -1]
    else:
        x = data
        y = None

    return Dataset(x, y)


def write_data_file(filename: str, dataset: Dataset, sep: str = ',', label: bool = False) -> None:
    """
    Writes a Dataset to a file without a header using numpy.savetxt.
    label: True to write the y as the last column.
    """
    if label and dataset.y is not None:
        data = np.column_stack((dataset.X, dataset.y))
    else:
        data = dataset.X

    np.savetxt(filename, data, delimiter=sep)