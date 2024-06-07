"""Transformer for data that contains Null values."""

import logging

import numpy as np
import pandas as pd

from rdt.errors import TransformerInputError

LOGGER = logging.getLogger(__name__)


class NullTransformer:
    """Transformer for data that contains Null values.

    Args:
        missing_value_replacement (object or None):
            Indicate what to do with the null values. If an integer, float or string is given,
            replace them with the given value. If the strings ``'mean'`` or ``'mode'`` are given,
            replace them with the corresponding aggregation (``'mean'`` only works for numerical
            values) if ``'random'`` replace each null value with a random value in the data range.
            If ``None`` is given, do not replace them. Defaults to ``None``.
        missing_value_generation (str or None):
            The way missing values are being handled. There are three strategies:

                * ``random``: Randomly generates missing values based on the percentage of
                  missing values.
                * ``from_column``: Creates a binary column that describes whether the original
                  value was missing. Then use it to recreate missing values.
                * ``None``: Do nothing with the missing values on the reverse transform. Simply
                  pass whatever data we get through.
    """

    nulls = None
    _missing_value_generation = None
    _missing_value_replacement = None
    _null_percentage = None

    def __init__(self, missing_value_replacement=None, missing_value_generation='random'):
        self._missing_value_replacement = missing_value_replacement
        if missing_value_generation not in (None, 'from_column', 'random'):
            raise TransformerInputError(
                "'missing_value_generation' must be one of the following values: "
                "None, 'from_column' or 'random'."
            )

        self._missing_value_generation = missing_value_generation
        self._min_value = None
        self._max_value = None

    def models_missing_values(self):
        """Indicate whether this transformer creates a null column on transform.

        Returns:
            bool:
                Whether a null column is created on transform.
        """
        return self._missing_value_generation == 'from_column'

    def _get_missing_value_replacement(self, data):
        """Get the fill value to use for the given data.

        Args:
            data (pd.Series):
                The data that is being transformed.

        Return:
            object:
                The fill value that needs to be used.

        Raise:
            TransformerInputError:
                Error raised when data only contains nans and ``_missing_value_replacement``
                is set to 'mean' or  'mode'.
        """
        if self._missing_value_replacement is None:
            return None

        if self._missing_value_replacement in {'mean', 'mode', 'random'} and pd.isna(data).all():
            msg = (
                f"'missing_value_replacement' cannot be set to '{self._missing_value_replacement}'"
                ' when the provided data only contains NaNs. Using 0 instead.'
            )
            LOGGER.info(msg)
            return 0

        if self._missing_value_replacement == 'mean':
            return data.mean()

        if self._missing_value_replacement == 'mode':
            return data.mode(dropna=True)[0]

        return self._missing_value_replacement

    def fit(self, data):
        """Fit the transformer to the data.

        Evaluate if the transformer has to create the null column or not.

        Args:
            data (pandas.Series):
                Data to transform.
        """
        self._missing_value_replacement = self._get_missing_value_replacement(data)
        if self._missing_value_replacement == 'random':
            self._min_value = data.min()
            self._max_value = data.max()

        if self._missing_value_generation is not None:
            null_values = data.isna().to_numpy()
            self.nulls = null_values.any()

            if not self.nulls and self.models_missing_values():
                self._missing_value_generation = None
                guidance_message = (
                    f'Guidance: There are no missing values in column {data.name}. '
                    'Extra column not created.'
                )
                LOGGER.info(guidance_message)

            if self._missing_value_generation == 'random':
                self._null_percentage = null_values.sum() / len(data)

    def _set_fitted_parameters(self, null_ratio):
        """Manually set the parameters on the transformer to get it into a fitted state.

        Args:
            null_ratio (float):
                The fraction of values to replace with null values.
        """
        if null_ratio < 0 or null_ratio > 1.0:
            raise ValueError('null_ratio should be a value between 0 and 1.')

        if null_ratio != 0:
            self.nulls = True
            self._null_percentage = null_ratio

    def transform(self, data):
        """Replace null values with the indicated ``missing_value_replacement``.

        If required, create the null indicator column.

        Args:
            data (pandas.Series or numpy.ndarray):
                Data to transform.

        Returns:
            numpy.ndarray
        """
        isna = data.isna()
        if self._missing_value_replacement == 'random':
            data_mask = list(
                np.random.uniform(low=self._min_value, high=self._max_value, size=len(data))
            )
            data = data.mask(data.isna(), data_mask)

        elif isna.any() and self._missing_value_replacement is not None:
            data = data.infer_objects().fillna(self._missing_value_replacement)

        if self._missing_value_generation == 'from_column':
            return pd.concat([data, isna.astype(np.float64)], axis=1).to_numpy()

        return data.to_numpy()

    def reverse_transform(self, data):
        """Restore null values to the data.

        If a null indicator column was created during fit, use it as a reference.
        Otherwise, randomly replace values with ``np.nan``. The percentage of values
        that will be replaced is the percentage of null values seen in the fitted data.

        Args:
            data (numpy.ndarray):
                Data to transform.

        Returns:
            pandas.Series
        """
        data = data.copy()
        if self._missing_value_generation == 'from_column':
            if self.nulls:
                isna = data[:, 1] > 0.5

            data = data[:, 0]

        elif self.nulls:
            isna = np.random.random((len(data),)) < self._null_percentage

        data = pd.Series(data)

        if self.nulls and isna.any():
            data.loc[isna] = np.nan

        return data
