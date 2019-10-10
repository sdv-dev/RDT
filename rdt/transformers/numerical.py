import numpy as np
import pandas as pd

from rdt.transformers.base import BaseTransformer
from rdt.transformers.null import NullTransformer


class NumericalTransformer(BaseTransformer):
    """Transformer for numerical data.

    The ``NumericalTransformer`` class allow transform and reverse of numerical values
    (``int`` and ``float``), and uses the ``NullTransformer`` to deal with null values.

    Args:
        dtype (data type):
            Data type of the data to transform. It will be used when reversing the transformation.
            If is not provided, will store the dtype of the data to be fitted.
            Defaults to None.

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

    def __init__(self, dtype=None, nan='mean', null_column=None):
        self.nan = nan
        self.null_column = null_column
        self.dtype = dtype

    def fit(self, data):
        """Prepare the transformer before convert data.

        Check if ``self.dtype`` is defined, if not use the data type from the data.
        Evaluate ``self.nan`` to get the fill value to instantiate the ``NullTransformer``.
        Finaly, create the ``null_transformer`` and fit the data.

        Args:
            data (pandas.Series or numpy.array):
                Data to fit.
        """
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
        """Transform numerical data.

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

        if self._dtype == np.int:
            data[pd.notnull(data)] = np.round(data[pd.notnull(data)]).astype(int)
            return data

        return data.astype(self._dtype)
