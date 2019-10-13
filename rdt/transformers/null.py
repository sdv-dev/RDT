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
        fill_value (object or None):
            Value to replace nulls. If ``None``, nans are not replaced.
        null_column (bool):
            Whether to create a new column to indicate which values were null or not.
            If ``None``, only create a new column when the data contains null values.
            If ``True``, always create the new column whether there are null values or not.
            If ``False``, do not create the new column.
            Defaults to ``None``.
        copy (bool):
            Whether to create a copy of the input data or modify it destructively.
    """

    def __init__(self, fill_value, null_column=None, copy=False):
        self.fill_value = fill_value
        self.null_column = null_column
        self.copy = copy

    def fit(self, data):
        """Fit the transformer to the data.

        Evaluate if the transformer has to create the null column or not.

        Args:
            data (pandas.Series or numpy.ndarray):
                Data to transform.
        """
        self.nulls = data.isnull().any()
        if self.null_column is None:
            self._null_column = self.nulls
        else:
            self._null_column = self.null_column

    def transform(self, data):
        """Replace null values with the indicated fill_value.

        If required, create the null indicator column.

        Args:
            data (pandas.Series or numpy.ndarray):
                Data to transform.

        Returns:
            numpy.ndarray
        """
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
        """Restore null values to the data.

        If a null indicator column was created dring fit, use it as a reference.
        Otherwise, replace all instances of ``fill_value`` that can be found in
        data.

        Args:
            data (numpy.ndarray):
                Data to transform.

        Returns:
            pandas.Series
        """
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
