import numpy as np
import pandas as pd

from rdt.transformers.base import BaseTransformer
from rdt.transformers.null import NullTransformer


class DatetimeTransformer(BaseTransformer):
    """Transformer for datetime data.

    The ``DatetimeTransformer`` class allow transform and reverse of datetime values,
    and uses the ``NullTransformer`` to deal with null values.

    Args:
        nan (str, int):
            Indicate how the ``NullTransformer`` will deal with null values.
            When ``nan`` is one of the following strings:
                - ``mean``: Compute the mean, dropping null values, to replace nulls.
                - ``mode``: Compute the mode, dropping null values, to replace nulls.
                - ``ignore``: Ignore null values, don't replace nothing.
            When ``nan`` is not any of the values above, then the replace value is ``nan``.
            Defaults to ``mean``.

        null_column (bool):
            Indicate when the ``NullTransformer`` have to create a new column with values
            in range 1 or 0 if the values are null or not respectively.
            When ``null_column`` is:
                - ``None``: Only create a new column when the data contains null values.
                - ``True``: Create always a new column even if the data don't contains null values.
                - ``False``: Never create a new column.
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
        """Prepare the transformer before convert data.

        First, transform datetimes into numeric values.
        Then, evaluate ``self.nan`` to get the fill value to instantiate the ``NullTransformer``.
        Finaly, create the ``null_transformer`` and fit the data.

        Args:
            data (pandas.Series or numpy.array):
                Data to fit.
        """
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
        self.null_transformer.fit(data)

    def transform(self, data):
        """Transform boolean data.

        First, transform datetimes into numeric values.

        Call the ``NullTransformer`` and return it's result.

        If the null transformer fill value  is already in the data and we don't
        create a null column, data can't be reversed. In this case we show a warning.

        Args:
            data (pandas.Series or numpy.array):
                Data to transform.

        Returns:
            numpy.array
        """
        if isinstance(data, np.ndarray):
            data = pd.Series(data)

        data = self._transform(data)

        return self.null_transformer.transform(data)

    def reverse_transform(self, data):
        """Converts data back into original format.

        Not all data is reversible. When the null transformer fill value is already in the
        original data and we haven't created the null column, data can't be reversed.

        Args:
            data (pandas.Series or numpy.array):
                Data to transform.

        Returns:
            pandas.Series
        """
        if self.nan != 'ignore':
            data = self.null_transformer.reverse_transform(data)

        data[pd.notnull(data)] = np.round(data[pd.notnull(data)]).astype(int)
        return pd.to_datetime(data)
