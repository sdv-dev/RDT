import time

import numpy as np
import pandas as pd

from rdt.transformers.base import BaseTransformer
from rdt.transformers.null import NullTransformer


class DateTimeTransformer(BaseTransformer):
    """Transformer for datetime data."""

    def __init__(self, nan='mean', null_column=True):
        self.nan = nan
        self.null_column = null_column
        self.null_transformer = None

    @staticmethod
    def _transform(datetimes):
        """Transform datetime values to integer."""
        nulls = datetimes.isnull()
        integers = datetimes.astype(int).astype(float).values
        integers[nulls] = np.nan

        return pd.Series(integers)

    def fit(self, data):
        if isinstance(data, np.ndarray):
            data = pd.Series(data)

        transformed = self._transform(data)

        if self.nan == 'mean':
            fill_value = transformed.mean()
        elif self.nan == 'mode':
            fill_value = transformed.mode(dropna=True)[0]
        elif self.nan == 'ignore':
            fill_value = None
        else:
            fill_value = self.nan

        self.null_transformer = NullTransformer(fill_value, self.null_column)

    def transform(self, data):
        if isinstance(data, np.ndarray):
            data = pd.Series(data)

        data = self._transform(data)

        return self.null_transformer.transform(data)

    def reverse_transform(self, data):
        if self.nan != 'ignore':
            data = self.null_transformer.reverse_transform(data)

        return pd.to_datetime(data)
