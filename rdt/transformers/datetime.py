"""Transformer for datetime data."""
import numpy as np
import pandas as pd

from rdt.transformers.base import BaseTransformer
from rdt.transformers.null import NullTransformer


class DatetimeTransformer(BaseTransformer):
    """Transformer for datetime data.

    This transformer replaces datetime values with an integer timestamp
    transformed to float.

    Null values are replaced using a ``NullTransformer``.

    Args:
        nan (int, str or None):
            Indicate what to do with the null values. If an integer is given, replace them
            with the given value. If the strings ``'mean'`` or ``'mode'`` are given, replace
            them with the corresponding aggregation. If ``None`` is given, do not replace them.
            Defaults to ``'mean'``.
        null_column (bool):
            Whether to create a new column to indicate which values were null or not.
            If ``None``, only create a new column when the data contains null values.
            If ``True``, always create the new column whether there are null values or not.
            If ``False``, do not create the new column.
            Defaults to ``None``.
        strip_constant (bool):
            Whether to optimize the output values by finding the smallest time unit that
            is not zero on the training datetimes and dividing the generated numerical
            values by the value of the next smallest time unit. This, a part from reducing the
            orders of magnitued of the transformed values, ensures that reverted values always
            are zero on the lower time units.
    """

    null_transformer = None
    divider = None

    def __init__(self, nan='mean', null_column=None, strip_constant=False):
        self.nan = nan
        self.null_column = null_column
        self.strip_constant = strip_constant

    def _find_divider(self, transformed):
        self.divider = 1
        multipliers = [10] * 9 + [60, 60, 24]
        for multiplier in multipliers:
            candidate = self.divider * multiplier
            if np.mod(transformed, candidate).any():
                break

            self.divider = candidate

    def _transform(self, datetimes):
        """Transform datetime values to integer."""
        nulls = datetimes.isnull()
        integers = np.zeros(len(datetimes))
        integers[~nulls] = datetimes[~nulls].astype(np.int64).astype(np.float64).values
        integers[nulls] = np.nan

        transformed = pd.Series(integers)
        if self.strip_constant:
            self._find_divider(transformed)
            transformed = transformed.floordiv(self.divider)

        return transformed

    def fit(self, data):
        """Fit the transformer to the data.

        Args:
            data (pandas.Series or numpy.ndarray):
                Data to fit the transformer to.
        """
        if isinstance(data, np.ndarray):
            data = pd.Series(data)

        transformed = self._transform(data)
        self.null_transformer = NullTransformer(self.nan, self.null_column, copy=True)
        self.null_transformer.fit(transformed)

    def transform(self, data):
        """Transform datetime values to float values.

        Args:
            data (pandas.Series or numpy.ndarray):
                Data to transform.

        Returns:
            numpy.ndarray
        """
        if isinstance(data, np.ndarray):
            data = pd.Series(data)

        data = self._transform(data)

        return self.null_transformer.transform(data)

    def reverse_transform(self, data):
        """Convert float values back to datetimes.

        Args:
            data (pandas.Series or numpy.ndarray):
                Data to transform.

        Returns:
            pandas.Series
        """
        if self.nan is not None:
            data = self.null_transformer.reverse_transform(data)

        if isinstance(data, np.ndarray) and (data.ndim == 2):
            data = data[:, 0]

        data[pd.notnull(data)] = np.round(data[pd.notnull(data)]).astype(np.int64)
        if self.strip_constant:
            data = data.astype(float) * self.divider

        return pd.to_datetime(data)
