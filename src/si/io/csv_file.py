import pandas as pd

from si.data.dataset import Dataset


def read_csv(filename: str, sep: str = ',', features: bool = True, label: bool = False) -> Dataset:
    """
    Reads a CSV file and returns a Dataset object.

    Parameters
    ----------
    filename: str
        Name/path of the file
    sep: str
        Value separator
    features: bool
        Whether the file has feature names (first row)
    label: bool
        Whether the file has y (assumed to be the last column)

    Returns
    -------
    Dataset
    """
    df = pd.read_csv(filename, sep=sep, header=0 if features else None)

    if label:
        x = df.iloc[:, :-1].to_numpy()
        y = df.iloc[:, -1].to_numpy()
        feature_names = list(df.columns[:-1]) if features else None
        label_name = str(df.columns[-1]) if features else None
    else:
        x = df.to_numpy()
        y = None
        feature_names = list(df.columns) if features else None
        label_name = None

    return Dataset(x, y, feature_names, label_name)


def write_csv(filename: str, dataset: Dataset, sep: str = ',', features: bool = True, label: bool = False) -> None:
    """
    Writes a Dataset object to a CSV file.

    Parameters
    ----------
    filename: str
        Name/path of the file
    dataset: Dataset
        Dataset to write
    sep: str
        Value separator
    features: bool
        Whether to write the header row with the feature names
    label: bool
        Whether to write y as the last column
    """
    df = pd.DataFrame(dataset.X, columns=dataset.features)

    if label and dataset.y is not None:
        label_name = dataset.label if dataset.label is not None else "y"
        df[label_name] = dataset.y

    df.to_csv(filename, sep=sep, index=False, header=features)
