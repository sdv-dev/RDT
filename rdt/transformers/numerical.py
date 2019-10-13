import numpy as np
import pandas as pd

from rdt.transformers.base import BaseTransformer
from rdt.transformers.null import NullTransformer


class NumericalTransformer(BaseTransformer):
    """Transformer for numerical data.

    This transformer replaces integer values with their float equivalent.
    Non null float values are not modified.

    Null values are replaced using a ``NullTransformer``.

    Args:
        dtype (data type):
            Data type of the data to transform. It will be used when reversing the
            transformation. If not provided, the dtype of the fit data will be used.
            Defaults to ``None``.
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

    def __init__(self, dtype=None, nan='mean', null_column=None):
        self.nan = nan
        self.null_column = null_column
        self.dtype = dtype

    def fit(self, data):
        """Fit the transformer to the data.

        Args:
            data (pandas.Series or numpy.ndarray):
                Data to fit.
        """
        if isinstance(data, np.ndarray):
            data = pd.Series(data)

        self._dtype = self.dtype or data.dtype

        if self.nan == 'mean':
            fill_value = data.mean()
        elif self.nan == 'mode':
            fill_value = data.mode(dropna=True)[0]
        else:
            fill_value = self.nan

        self.null_transformer = NullTransformer(fill_value, self.null_column)
        self.null_transformer.fit(data)

    def transform(self, data):
        """Transform numerical data.

        Integer values are replaced by their float equivalent. Non null float values
        are left unmodified.

        Args:
            data (pandas.Series or numpy.ndarray):
                Data to transform.

        Returns:
            numpy.ndarray
        """
        if isinstance(data, np.ndarray):
            data = pd.Series(data)

        return self.null_transformer.transform(data)

    def reverse_transform(self, data):
        """Converts data back into original format.

        Args:
            data (numpy.ndarray):
                Data to transform.

        Returns:
            pandas.Series
        """
        if self.nan is not None:
            data = self.null_transformer.reverse_transform(data)

        if self._dtype == np.int:
            data[pd.notnull(data)] = np.round(data[pd.notnull(data)]).astype(int)
            return data

        return data.astype(self._dtype)
