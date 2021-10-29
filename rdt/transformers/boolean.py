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

    INPUT_TYPE = 'boolean'
    DETERMINISTIC_TRANSFORM = True
    DETERMINISTIC_REVERSE = True

    null_transformer = None

    def __init__(self, nan=-1, null_column=None):
        self.nan = nan
        self.null_column = null_column

    def get_output_types(self):
        """Return the output types returned by this transformer.

        Returns:
            dict:
                Mapping from the transformed column names to the produced data types.
        """
        output_types = {
            'value': 'float',
        }
        if self.null_transformer and self.null_transformer.creates_null_column():
            output_types['is_null'] = 'float'

        return self._add_prefix(output_types)

    def _fit(self, data):
        """Fit the transformer to the data.

        Args:
            data (pandas.Series):
                Data to fit to.
        """
        self.null_transformer = NullTransformer(self.nan, self.null_column, copy=True)
        self.null_transformer.fit(data)

    def _transform(self, data):
        """Transform boolean to float.

        The boolean values will be replaced by the corresponding integer
        representations as float values.

        Args:
            data (pandas.Series):
                Data to transform.

        Returns
            pandas.DataFrame or pandas.Series
        """
        data = pd.to_numeric(data, errors='coerce')
        return self.null_transformer.transform(data).astype(float)

    def _reverse_transform(self, data):
        """Transform float values back to the original boolean values.

        Args:
            data (pandas.DataFrame or pandas.Series):
                Data to revert.

        Returns:
            pandas.Series:
                Reverted data.
        """
        if not isinstance(data, np.ndarray):
            data = data.to_numpy()

        if self.nan is not None:
            data = self.null_transformer.reverse_transform(data)

        if isinstance(data, np.ndarray):
            if data.ndim == 2:
                data = data[:, 0]

            data = pd.Series(data)

        return np.round(data).clip(0, 1).astype('boolean').astype('object')
