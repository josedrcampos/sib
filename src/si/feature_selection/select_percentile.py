from typing import Callable

import numpy as np

from si.base.transformer import Transformer
from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification


class SelectPercentile(Transformer):
    """
    Selects a given percentage of the features, the ones with the highest F values.
    Ties at the threshold are resolved so that exactly that percentage is selected.
    """

    def __init__(self, score_func: Callable = f_classification, percentile: float = 50, **kwargs):
        """
        Parameters
        ----------
        score_func: Callable
            Function that receives a Dataset and returns (F, p) for every feature.
        percentile: float
            Percentage of features to select, between 0 and 100.
        """
        super().__init__(**kwargs)
        if not 0 <= percentile <= 100:
            raise ValueError("percentile must be between 0 and 100")
        self.score_func = score_func
        self.percentile = percentile

        # estimated parameters
        self.F = None
        self.p = None

    def _fit(self, dataset: Dataset) -> 'SelectPercentile':
        """
        Estimates the F and p values of every feature with score_func.
        """
        self.F, self.p = self.score_func(dataset)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Keeps the given percentage of features with the highest F values.
        """
        # constant features give F = NaN; F is always >= 0, so -1 makes them the worst score
        F = np.where(np.isnan(self.F), -1.0, self.F)
        n_features = len(F)
        n_select = int(n_features * self.percentile / 100)

        if n_select == 0:
            mask = np.zeros(n_features, dtype=bool)
        else:
            # 1. threshold = F value at the (100 - percentile) percentile
            threshold = np.percentile(F, 100 - self.percentile)

            # 2. features strictly above the threshold are always selected
            mask = F > threshold

            # 3. if that is not enough, add the features tied at the threshold,
            #    in their original order, until n_select is reached
            missing = n_select - mask.sum()
            if missing > 0:
                tied = np.where(F == threshold)[0]
                mask[tied[:missing]] = True

            # safety net: never select more than n_select
            # (keeps the n_select highest F values among the selected ones)
            if mask.sum() > n_select:
                selected = np.where(mask)[0]
                keep = selected[np.argsort(F[selected])[-n_select:]]
                mask = np.zeros(n_features, dtype=bool)
                mask[keep] = True

        features = np.array(dataset.features)[mask].tolist()
        return Dataset(X=dataset.X[:, mask], y=dataset.y, features=features, label=dataset.label)


if __name__ == '__main__':
    # example from the slides: 10 features, percentile = 40 -> 4 features
    F = np.array([1.2, 3.4, 2.1, 5.6, 4.3, 5.6, 7.8, 6.5, 5.6, 3.2])
    dataset = Dataset(X=np.random.rand(5, 10), y=np.array([0, 1, 0, 1, 0]))

    selector = SelectPercentile(score_func=lambda ds: (F, np.zeros_like(F)), percentile=40)
    new_dataset = selector.fit_transform(dataset)
    print(new_dataset.features)          # ['feat_3', 'feat_5', 'feat_6', 'feat_7']
    print(F[[3, 5, 6, 7]])               # [5.6 5.6 7.8 6.5]
