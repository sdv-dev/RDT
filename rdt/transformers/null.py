"""Transformer for data that contains Null values."""

import warnings

import numpy as np
import pandas as pd

IRREVERSIBLE_WARNING = (
    'Replacing nulls with existing value without `null_column`, which is not reversible. '
    'Use `null_column=True` to ensure that the transformation is reversible.'
)


class NullTransformer():
    """Transformer for data that contains Null values.

    Args:
        fill_value (object or None):
            Value to replace nulls. If ``None``, nans are not replaced.
        null_column (bool):
            Whether to create a new column to indicate which values were null or not.
            If ``None``, only create a new column when the data contains null values.
            If ``True``, always create the new column whether there are null values or not.
            If ``False``, do not create the new column.
            Defaults to ``None``.
        copy (bool):
            Whether to create a copy of the input data or modify it destructively.
    """

    nulls = None
    _null_column = None
    _fill_value = None

    def __init__(self, fill_value, null_column=None, copy=False):
        self.fill_value = fill_value
        self.null_column = null_column
        self.copy = copy

    def creates_null_column(self):
        """Indicate whether this transformer creates a null column on transform.

        Returns:
            bool:
                Whether a null column is created on transform.
        """
        return bool(self._null_column)

    def fit(self, data):
        """Fit the transformer to the data.

        Evaluate if the transformer has to create the null column or not.

        Args:
            data (pandas.Series or numpy.ndarray):
                Data to transform.
        """
        null_values = data.isna().to_numpy()
        self.nulls = null_values.any()
        contains_not_null = not null_values.all()
        if self.fill_value == 'mean':
            self._fill_value = data.mean() if contains_not_null else 0
        elif self.fill_value == 'mode':
            # If there are multiple values with the same frequency, select the smallest value
            self._fill_value = data.mode(dropna=True)[0] if contains_not_null else 0
        else:
            self._fill_value = self.fill_value

        if self.null_column is None:
            self._null_column = self.nulls
        else:
            self._null_column = self.null_column

    def transform(self, data):
        """Replace null values with the indicated fill_value.

        If required, create the null indicator column.

        Args:
            data (pandas.Series or numpy.ndarray):
                Data to transform.

        Returns:
            numpy.ndarray
        """
        if self._null_column or (self.nulls and self._fill_value is not None):
            isna = data.isna()
            if not isinstance(data, pd.Series):
                data = pd.Series(data)

<<<<<<< HEAD
        if not self._null_column and (self._fill_value == data.array).any():
            warnings.warn(IRREVERSIBLE_WARNING)

        if self.nulls and self._fill_value is not None:
            if not self.copy:
                data[isna] = self._fill_value
            else:
                data = data.fillna(self._fill_value)

        if self._null_column:
            return pd.concat([data, isna.astype('int')], axis=1).to_numpy()
=======
            isna = data.isna()
            if self.nulls and self._fill_value is not None:
                if not self.copy:
                    data[isna] = self._fill_value
                else:
                    data = data.fillna(self._fill_value)

            if self._null_column:
                return pd.concat([data, isna.astype('int')], axis=1).to_numpy()

            if self._fill_value in data:
                warnings.warn(IRREVERSIBLE_WARNING)
>>>>>>> v0.6.0-dev

        return data.array

    def reverse_transform(self, data):
        """Restore null values to the data.

        If a null indicator column was created during fit, use it as a reference.
        Otherwise, replace all instances of ``fill_value`` that can be found in
        data.

        Args:
            data (numpy.ndarray):
                Data to transform.

        Returns:
            pandas.Series
        """
        if self.nulls:
            if self._null_column:
                isna = data[:, 1] > 0.5
                data = pd.Series(data[:, 0])
            else:
                isna = np.where(self._fill_value == data)[0]
                data = pd.Series(data)

<<<<<<< HEAD
            if len(isna) > 0:
=======
            if isna.any():
>>>>>>> v0.6.0-dev
                if self.copy:
                    data = data.copy()

                data.iloc[isna] = np.nan

        return data
