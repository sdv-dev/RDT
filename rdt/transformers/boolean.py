"""Transformer for boolean data."""

import numpy as np
import pandas as pd

from rdt.transformers.base import BaseTransformer
from rdt.transformers.null import NullTransformer


class BinaryEncoder(BaseTransformer):
    """Transformer for boolean data.

    This transformer replaces boolean values with their integer representation
    transformed to float.

    Null values are replaced using a ``NullTransformer``.

    Args:
        missing_value_replacement (object):
            Indicate what to replace the null values with. If the string ``'mode'`` is given,
            replace them with the most common value.
            Defaults to ``mode``.
        model_missing_values (bool):
            Whether to create a new column to indicate which values were null or not. The column
            will be created only if there are null values. If ``True``, create the new column if
            there are null values. If ``False``, do not create the new column even if there
            are null values. Defaults to ``False``.
    """

    INPUT_SDTYPE = 'boolean'
    null_transformer = None

    def __init__(self, missing_value_replacement='mode', model_missing_values=False):
        super().__init__()
        self._set_missing_value_replacement('mode', missing_value_replacement)
        self.model_missing_values = model_missing_values

    def _fit(self, data):
        """Fit the transformer to the data.

        Args:
            data (pandas.Series):
                Data to fit to.
        """
        self.null_transformer = NullTransformer(
            self.missing_value_replacement,
            self.model_missing_values
        )
        self.null_transformer.fit(data)
        if self.null_transformer.models_missing_values():
            self.output_properties['is_null'] = {'sdtype': 'float', 'next_transformer': None}

    def _transform(self, data):
        """Transform boolean to float.

        The boolean values will be replaced by the corresponding integer
        representations as float values.

        Args:
            data (pandas.Series):
                Data to transform.

        Returns
            np.ndarray
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

        data = self.null_transformer.reverse_transform(data)
        if isinstance(data, np.ndarray):
            if data.ndim == 2:
                data = data[:, 0]

            data = pd.Series(data)

        isna = data.isna()
        data = np.round(data).clip(0, 1).astype('boolean').astype('object')
        data[isna] = np.nan

        return data
