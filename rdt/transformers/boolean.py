"""Transformer for boolean data."""

import numpy as np
import pandas as pd

from rdt.transformers.base import BaseTransformer
from rdt.transformers.null import NullTransformer


class BooleanTransformer(BaseTransformer):
    """Transformer for boolean data.

    This transformer replaces boolean values with their integer representation
    transformed to float.

    Null values are replaced using a ``NullTransformer``.

    Args:
        nan (int or None):
            Replace null values with the given value. If ``None``, do not replace them.
            Defaults to ``-1``.
        null_column (bool):
            Whether to create a new column to indicate which values were null or not.
            If ``None``, only create a new column when the fit data contains null values.
            If ``True``, always create the new column whether there are null values or not.
            If ``False``, do not create the new column.
            Defaults to ``None``.
    """

    null_transformer = None

    def __init__(self, nan=-1, null_column=None):
        self.nan = nan
        self.null_column = null_column

    def fit(self, data):
        """Fit the transformer to the data.

        Args:
            data (pandas.Series or numpy.ndarray):
                Data to fit to.
        """
        if isinstance(data, np.ndarray):
            data = pd.Series(data)

        self.null_transformer = NullTransformer(self.nan, self.null_column)
        self.null_transformer.fit(data)

    def transform(self, data):
        """Transform boolean to float.

        The boolean values will be replaced by the corresponding integer
        representations as float values.

        Args:
            data (pandas.Series or numpy.ndarray):
                Data to transform.

        Returns
            numpy.ndarray
        """
        if isinstance(data, np.ndarray):
            data = pd.Series(data)

        data.loc[data.notnull()] = data.dropna().astype(int)

        return self.null_transformer.transform(data).astype(float)

    def reverse_transform(self, data):
        """Transform float values back to the original boolean values.

        Args:
            data (numpy.ndarray):
                Data to revert.

        Returns:
            pandas.Series
        """
        if self.nan is not None:
            data = self.null_transformer.reverse_transform(data)

        if isinstance(data, np.ndarray):
            if data.ndim == 2:
                data = data[:, 0]

            data = pd.Series(data)

        data[pd.notnull(data)] = np.round(data[pd.notnull(data)]).astype(bool)
        return data
