import numpy as np
import pandas as pd

from rdt.transformers.base import BaseTransformer
from rdt.transformers.null import NullTransformer


class BooleanTransformer(BaseTransformer):
    """Transformer for boolean data.

    The ``BooleanTransformer`` class allow transform and reverse of boolean values,
    and uses the ``NullTransformer`` to deal with null values.

    Args:
        nan (str or int):
            Indicates how the ``NullTransformer`` will deal with null values.
            When ``nan`` is a string equal to ``ignore``, null values will not be replaces.
            Otherwise the replace value is ``nan``.
            Defaults to ``-1``.

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

    def __init__(self, nan=-1, null_column=None):
        self.nan = nan
        self.null_column = null_column

    def fit(self, data):
        """Prepare the transformer before convert data.

        Evaluate ``self.nan`` to get the fill value to instantiate the ``NullTransformer``
        and create the ``null_transformer`` and fit the data.

        Args:
            data (pandas.Series or numpy.array):
                Data to fit.
        """
        if isinstance(data, np.ndarray):
            data = pd.Series(data)

        if self.nan == 'ignore':
            fill_value = None
        else:
            fill_value = self.nan

        self.null_transformer = NullTransformer(fill_value, self.null_column)
        self.null_transformer.fit(data)

    def transform(self, data):
        """Transform boolean data.

        Transform boolean data into numeric values dropping nulls.
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

        data.loc[data.notnull()] = data.dropna().astype(int)

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

        data[pd.notnull(data)] = np.round(data[pd.notnull(data)]).astype(bool)
        return data
