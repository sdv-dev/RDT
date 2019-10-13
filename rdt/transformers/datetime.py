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
    """

    null_transformer = None

    def __init__(self, nan='mean', null_column=None):
        self.nan = nan
        self.null_column = null_column

    @staticmethod
    def _transform(datetimes):
        """Transform datetime values to integer."""
        nulls = datetimes.isnull()
        integers = datetimes.astype(int).astype(float).values
        integers[nulls] = np.nan

        return pd.Series(integers)

    def fit(self, data):
        """Fit the transformer to the data.

        Args:
            data (pandas.Series or numpy.ndarray):
                Data to fit the transformer to.
        """
        if isinstance(data, np.ndarray):
            data = pd.Series(data)

        transformed = self._transform(data)

        if self.nan == 'mean':
            fill_value = transformed.mean()
        elif self.nan == 'mode':
            fill_value = transformed.mode(dropna=True)[0]
        else:
            fill_value = self.nan

        self.null_transformer = NullTransformer(fill_value, self.null_column)
        self.null_transformer.fit(data)

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

        data[pd.notnull(data)] = np.round(data[pd.notnull(data)]).astype(int)
        return pd.to_datetime(data)
