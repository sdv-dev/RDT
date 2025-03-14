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
            **DEPRECATED** Whether to create a new column to indicate which values were null or
            not. The column will be created only if there are null values. If ``True``, create
            the new column if there are null values. If ``False``, do not create the new column
            even if there are null values. Defaults to ``False``.
        missing_value_generation (str or None):
            The way missing values are being handled. There are three strategies:

                * ``random``: Randomly generates missing values based on the percentage of
                  missing values.
                * ``from_column``: Creates a binary column that describes whether the original
                  value was missing. Then use it to recreate missing values.
                * ``None``: Do nothing with the missing values on the reverse transform. Simply
                  pass whatever data we get through.
    """

    INPUT_SDTYPE = 'boolean'
    null_transformer = None

    def __init__(
        self,
        missing_value_replacement='mode',
        model_missing_values=None,
        missing_value_generation='random',
    ):
        super().__init__()
        self.missing_value_replacement = missing_value_replacement
        self._set_missing_value_generation(missing_value_generation)
        if model_missing_values is not None:
            self._set_model_missing_values(model_missing_values)

    def _fit(self, data):
        """Fit the transformer to the data.

        Args:
            data (pandas.Series):
                Data to fit to.
        """
        self.null_transformer = NullTransformer(
            self.missing_value_replacement, self.missing_value_generation
        )
        self.null_transformer.fit(data)
        if self.null_transformer.models_missing_values():
            self.output_properties['is_null'] = {
                'sdtype': 'float',
                'next_transformer': None,
            }

    def _transform(self, data):
        """Transform boolean to float.

        The boolean values will be replaced by the corresponding integer
        representations as float values.

        Args:
            data (pandas.Series):
                Data to transform.

        Returns:
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

    def _set_fitted_parameters(self, column_name, null_transformer):
        """Manually set the parameters on the transformer to get it into a fitted state.

        Args:
            column_name (str):
                The name of the column to use for the transformer.
            null_transformer (NullTransformer):
                A fitted null transformer instance that can be used to generate
                null values for the column.
        """
        self.reset_randomization()
        self.columns = [column_name]
        self.output_columns = [column_name]
        self.null_transformer = null_transformer
        if self.null_transformer.models_missing_values():
            self.output_columns.append(column_name + '.is_null')
