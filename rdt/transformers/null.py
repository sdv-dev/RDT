import pandas as pd
import numpy as np
import warnings

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

    def transform(self, data):
        if isinstance(data, np.ndarray):
            data = pd.Series(data)

        _isnull = data.isnull()
        if _isnull.any() and self.fill_value is not None:
            if not self.copy:
                data[_isnull] = self.fill_value
            else:
                data = data.fillna(self.fill_value)

        if (self.null_column is None and _isnull.any()) or self.null_column:
            return pd.concat([data, _isnull.astype('int')], axis=1).values

        elif self.fill_value in data:
            warnings.warn(IRREVERSIBLE_WARNING)

        return data.values

    def reverse_transform(self, data):
        shape = data.shape
        if self.null_column and len(shape) == 2 and shape[1] == 2:
            _isnull = data[:, 1].astype('bool')
            data = pd.Series(data[:, 0])
        else:
            _isnull = data == self.fill_value
            data = pd.Series(data)

        if self.copy:
            data = data.copy()

        data.iloc[_isnull] = np.nan
        return data
