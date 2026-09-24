import numpy as np

from si.base.transformer import Transformer
from si.data.dataset import Dataset


class VarianceThreshold(Transformer):
    def __init__(self, threshold: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        if threshold < 0:
            raise ValueError("Threshold must be a non-negative value")
        self.threshold = threshold

        # estimated parameters
        self.variance = None

    def _fit(self, dataset: Dataset):
        self.variance = np.var(dataset.X, axis=0)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        mask = self.variance > self.threshold
        new_X = dataset.X[:, mask]
        new_features = np.array(dataset.features)[mask].tolist()
        return Dataset(X=new_X, y=dataset.y, features=new_features, label=dataset.label)


if __name__ == "__main__":
    X = np.array([[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]])
    dataset = Dataset(X, features=["f0", "f1", "f2", "f3"])

    vt = VarianceThreshold(threshold=1.0)
    new_dataset = vt.fit_transform(dataset)
    print("original variance:", vt.variance)
    print("selected features:", new_dataset.features)
    print("new X:\n", new_dataset.X)
