import pandas as pd
import numpy as np

from rdt.transformers.base import BaseTransformer


class NumericalTransformer(BaseTransformer):
    """Transformer for numerical data."""

    default = None

    def fit(self, data):
        if isinstance(data, np.ndarray):
            data = pd.Series(data)

        self.dtype = data.dtype
        self.default = data.mean()

    def transform(self, data):
        if isinstance(data, np.ndarray):
            data = pd.Series(data)

        transformed = data.fillna(self.default)

        return transformed.to_numpy()

    def reverse_transform(self, data):
        if not isinstance(data, np.ndarray):
            raise ValueError

        return data.astype(self.dtype)
