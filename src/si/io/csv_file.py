import pandas as pd

from si.data.dataset import Dataset


def read_csv(filename: str, sep: str = ',', features: bool = True, label: bool = False) -> Dataset:
    """
    Reads a CSV file and returns a Dataset.

    The first row of a CSV file is always treated as the header (column names).
    Files without a header are handled by si.io.data_file (read_data_file).

    Parameters
    ----------
    filename: str
        Name/path of the file
    sep: str
        Value separator
    features: bool
        True if the file has feature columns whose names should be kept.
        If False, the feature names are not used (generic names are generated).
    label: bool
        True if the last column is the dependent variable (y).

    Returns
    -------
    Dataset
    """
    df = pd.read_csv(filename, sep=sep, header=0)

    if label:
        x = df.iloc[:, :-1].to_numpy()
        y = df.iloc[:, -1].to_numpy()
        feature_names = list(df.columns[:-1]) if features else None
        label_name = str(df.columns[-1])
    else:
        x = df.to_numpy()
        y = None
        feature_names = list(df.columns) if features else None
        label_name = None

    return Dataset(x, y, feature_names, label_name)


def write_csv(filename: str, dataset: Dataset, sep: str = ',', features: bool = True, label: bool = False) -> None:
    """
    Writes a Dataset to a CSV file.

    The header row is always written, so that read_csv can read the file back
    (read_csv always treats the first row as the header).

    Parameters
    ----------
    filename: str
        Name/path of the file
    dataset: Dataset
        The dataset to write
    sep: str
        Value separator
    features: bool
        True to write the real feature names; False to write generic names (feat_0, feat_1, ...).
    label: bool
        True to write y as the last column.
    """
    if features:
        columns = list(dataset.features)
    else:
        columns = [f"feat_{i}" for i in range(dataset.X.shape[1])]
    df = pd.DataFrame(dataset.X, columns=columns)

    if label and dataset.y is not None:
        label_name = dataset.label if dataset.label is not None else "y"
        df[label_name] = dataset.y

    df.to_csv(filename, sep=sep, index=False, header=True)
