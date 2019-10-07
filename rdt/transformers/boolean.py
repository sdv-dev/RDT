import numpy as np
import pandas as pd

from rdt.transformers.base import BaseTransformer
from rdt.transformers.null import NullTransformer


class BooleanTransformer(BaseTransformer):
    """Transformer for boolean data."""

    null_transformer = None

    def __init__(self, nan=-1, null_column=None):
        self.nan = nan
        self.null_column = null_column

    def fit(self, data):
        if isinstance(data, np.ndarray):
            data = pd.Series(data)

        if self.nan == 'ignore':
            fill_value = None
        else:
            fill_value = self.nan

        self.null_transformer = NullTransformer(fill_value, self.null_column)
        self.null_transformer.fit(data)

    def transform(self, data):
        if isinstance(data, np.ndarray):
            data = pd.Series(data)

        data.loc[data.notnull()] = data.dropna().astype(int)

        return self.null_transformer.transform(data)

    def reverse_transform(self, data):
        if self.nan != 'ignore':
            data = self.null_transformer.reverse_transform(data)

        data.loc[data.notnull()] = data.dropna().round().astype(bool)
        return data
