import warnings

import numpy as np
import pandas as pd

from rdt.transformers.base import BaseTransformer

IRREVERSIBLE_WARNING = (
    'Replacing nulls with existing value without `null_column`, which is not reversible. '
    'Use `null_column=True` to ensure that the transformation is reversible.'
)


class NullTransformer(BaseTransformer):
    """Transformer for null data.

    Args:
        fill_value:
            Value to replace nulls. Not used if `None`.
        null_column (bool or None):
            If `True`, always create a column indicating whether
            each value is null or not. If `None`, create it only
            if there is at least one null value. If `False`, never
            create it, even if there are null values.
        copy (bool):
            Whether to create a copy of the input data or modify it
            destructively.
    """

    def __init__(self, fill_value, null_column=None, copy=False):
        self.fill_value = fill_value
        self.null_column = null_column
        self.copy = copy

    def fit(self, data):
        self.nulls = data.isnull().any()
        if self.null_column is None:
            self._null_column = self.nulls
        else:
            self._null_column = self.null_column

    def transform(self, data):
        if self.nulls:
            isnull = data.isnull()
            if self.nulls and self.fill_value is not None:
                if not self.copy:
                    data[isnull] = self.fill_value
                else:
                    data = data.fillna(self.fill_value)

            if self._null_column:
                return pd.concat([data, isnull.astype('int')], axis=1).values

            elif self.fill_value in data.values:
                warnings.warn(IRREVERSIBLE_WARNING)

        return data.values

    def reverse_transform(self, data):
        if self.nulls:
            if self._null_column:
                isnull = data[:, 1] > 0.5
                data = pd.Series(data[:, 0])
            else:
                isnull = np.where(self.fill_value == data)[0]
                data = pd.Series(data)

            if isnull.any():
                data.iloc[isnull] = np.nan

        return data
