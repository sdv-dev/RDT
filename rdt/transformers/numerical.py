import numpy as np
import pandas as pd

from rdt.transformers.base import BaseTransformer
from rdt.transformers.null import NullTransformer


class NumericalTransformer(BaseTransformer):
    """Transformer for numerical data."""

    null_transformer = None

    def __init__(self, dtype=None, nan='mean', null_column=None):
        self.nan = nan
        self.null_column = null_column
        self.dtype = dtype

    def fit(self, data):
        if isinstance(data, np.ndarray):
            data = pd.Series(data)

        self._dtype = self.dtype or data.dtype

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
        if isinstance(data, np.ndarray):
            data = pd.Series(data)

        if self.nan != 'ignore':
            data = self.null_transformer.reverse_transform(data)

        if self._dtype == np.int:
            data.loc[data.notnull()] = data.dropna().round().astype(int)
            return data

        return data.astype(self._dtype)
