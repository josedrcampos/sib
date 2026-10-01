import numpy as np

from si.data.dataset import Dataset


def read_data_file(filename: str, sep: str = ',', label: bool = False) -> Dataset:
    """
    Reads a headerless data file with numpy.genfromtxt and returns a Dataset object.

    Parameters
    ----------
    filename: str
        Name/path of the file
    sep: str
        Value separator
    label: bool
        Whether the file has y (assumed to be the last column)

    Returns
    -------
    Dataset
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
    Writes a Dataset object to a headerless file with numpy.savetxt.

    Parameters
    ----------
    filename: str
        Name/path of the file
    dataset: Dataset
        Dataset to write
    sep: str
        Value separator
    label: bool
        Whether to write y as the last column
    """
    if label and dataset.y is not None:
        data = np.column_stack((dataset.X, dataset.y))
    else:
        data = dataset.X

    np.savetxt(filename, data, delimiter=sep)
