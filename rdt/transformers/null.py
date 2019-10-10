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

    The ``NullTransformer`` class allow replace null values with a given value
    and create a null column if needed.

    Args:
        fill_value (int):
            Value to replace nulls. Not used if `None`.
        null_column (bool):
            When ``null_column`` is:
                - ``None``: Only create a new column when the data contains null values.
                - ``True``: Create always a new column even if the data don't contains null values.
                - ``False``: Never create a new column.
            Defaults to ``None``.
        copy (bool):
            Whether to create a copy of the input data or modify it destructively.
    """

    def __init__(self, fill_value, null_column=None, copy=False):
        self.fill_value = fill_value
        self.null_column = null_column
        self.copy = copy

    def fit(self, data):
        """Prepare the transformer to convert data.

        Evaluate when the transformer have to create the null column or not.

        Args:
            data (pandas.Series or numpy.array):
                Data to transform.
        """
        self.nulls = data.isnull().any()
        if self.null_column is None:
            self._null_column = self.nulls
        else:
            self._null_column = self.null_column

    def transform(self, data):
        """Transform null data.

        When the data have null values, evaluate if should replace null values
        and/or create the null column.

        Args:
            data (pandas.Series or numpy.array):
                Data to transform.

        Returns:
            numpy.array
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
        """Converts data back into original format.

        If the ``fill_value`` was in ``data`` when transforming the original
        data can't be restored.

        Args:
            data (pandas.Series or numpy.array):
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
