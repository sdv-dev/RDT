import numpy as np
import pandas as pd

from rdt.transformers.base import BaseTransformer
from rdt.transformers.null import NullTransformer


class NumericalTransformer(BaseTransformer):
    """Transformer for numerical data."""

    null_transformer = None
    dtype = None

    def __init__(self, nan='mean', null_column=None):
        self.nan = nan
        self.null_column = null_column

    def fit(self, data):
        if isinstance(data, np.ndarray):
            data = pd.Series(data)

        self.dtype = data.dtype

        if self.nan == 'mean':
            fill_value = data.mean()
        elif self.nan == 'mode':
            fill_value = data.mode(dropna=True)[0]
        elif self.nan == 'ignore':
            fill_value = None
        else:
            fill_value = self.nan

        self.null_transformer = NullTransformer(fill_value, self.null_column)
        self.null_transformer.fit(data)

    def transform(self, data):
        if isinstance(data, np.ndarray):
            data = pd.Series(data)

        return self.null_transformer.transform(data)

    def reverse_transform(self, data):
        if self.nan != 'ignore':
            data = self.null_transformer.reverse_transform(data)

        if self.dtype == np.int:
            return np.round(data).astype(self.dtype)

        return data.astype(self.dtype)
