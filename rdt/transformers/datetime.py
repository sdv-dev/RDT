"""Transformer for datetime data."""
import numpy as np
import pandas as pd

from rdt.transformers.base import BaseTransformer
from rdt.transformers.null import NullTransformer


class DatetimeTransformer(BaseTransformer):
    """Transformer for datetime data.

    This transformer replaces datetime values with an integer timestamp
    transformed to float.

    Null values are replaced using a ``NullTransformer``.

    Args:
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
        strip_constant (bool):
            Whether to optimize the output values by finding the smallest time unit that
            is not zero on the training datetimes and dividing the generated numerical
            values by the value of the next smallest time unit. This, a part from reducing the
            orders of magnitued of the transformed values, ensures that reverted values always
            are zero on the lower time units.
        format (str):
            The strftime to use for parsing time. For more information, see
            https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior.
    """

    INPUT_TYPE = 'datetime'
    DETERMINISTIC_TRANSFORM = True
    DETERMINISTIC_REVERSE = True
    COMPOSITION_IS_IDENTITY = True

    null_transformer = None
    divider = None

    def __init__(self, nan='mean', null_column=None, strip_constant=False, datetime_format=None):
        self.nan = nan
        self.null_column = null_column
        self.strip_constant = strip_constant
        self.datetime_format = datetime_format

    def is_composition_identity(self):
        """Return whether composition of transform and reverse transform produces the input data.

        Returns:
            bool:
                Whether or not transforming and then reverse transforming returns the input data.
        """
        if self.null_transformer and not self.null_transformer.creates_null_column():
            return False

        return self.COMPOSITION_IS_IDENTITY

    def get_output_types(self):
        """Return the output types supported by the transformer.

        Returns:
            dict:
                Mapping from the transformed column names to supported data types.
        """
        output_types = {
            'value': 'float',
        }
        if self.null_transformer and self.null_transformer.creates_null_column():
            output_types['is_null'] = 'float'

        return self._add_prefix(output_types)

    def _find_divider(self, transformed):
        self.divider = 1
        multipliers = [10] * 9 + [60, 60, 24]
        for multiplier in multipliers:
            candidate = self.divider * multiplier
            if (transformed % candidate).any():
                break

            self.divider = candidate

    def _convert_to_datetime(self, data):
        if data.dtype == 'object':
            try:
                data = pd.to_datetime(data, format=self.datetime_format)

            except ValueError as error:
                if 'Unknown string format:' in str(error):
                    message = 'Data must be of dtype datetime, or castable to datetime.'
                    raise TypeError(message) from None

                raise ValueError('Data does not match specified datetime format.') from None

        return data

    def _transform_helper(self, datetimes):
        """Transform datetime values to integer."""
        datetimes = self._convert_to_datetime(datetimes)
        nulls = datetimes.isna()
        integers = pd.to_numeric(datetimes, errors='coerce').to_numpy().astype(np.float64)
        integers[nulls] = np.nan
        transformed = pd.Series(integers)

        if self.strip_constant:
            self._find_divider(transformed)
            transformed = transformed // self.divider

        return transformed

    def _fit(self, data):
        """Fit the transformer to the data.

        Args:
            data (pandas.Series):
                Data to fit the transformer to.
        """
        transformed = self._transform_helper(data)
        self.null_transformer = NullTransformer(self.nan, self.null_column, copy=True)
        self.null_transformer.fit(transformed)

    def _transform(self, data):
        """Transform datetime values to float values.

        Args:
            data (pandas.Series):
                Data to transform.

        Returns:
            numpy.ndarray
        """
        data = self._transform_helper(data)
        return self.null_transformer.transform(data)

    def _reverse_transform(self, data):
        """Convert float values back to datetimes.

        Args:
            data (pandas.Series or numpy.ndarray):
                Data to transform.

        Returns:
            pandas.Series
        """
        if not isinstance(data, np.ndarray):
            data = data.to_numpy()

        if self.nan is not None:
            data = self.null_transformer.reverse_transform(data)

        if isinstance(data, np.ndarray) and (data.ndim == 2):
            data = data[:, 0]

        data = np.round(data.astype(np.float64))
        if self.strip_constant:
            data = data * self.divider

        return pd.to_datetime(data)


class DatetimeRoundedTransformer(DatetimeTransformer):
    """Transformer for datetime data.

    This transformer replaces datetime values with an integer timestamp transformed to float.
    It optimizes the output values by finding the smallest time unit that is not zero on
    the training datetimes and dividing the generated numerical values by the value of the next
    smallest time unit. This, apart from reducing the orders of magnitued of the transformed
    values, ensures that reverted values always are zero on the lower time units.

    Null values are replaced using a ``NullTransformer``.

    This class behaves exactly as the ``DatetimeTransformer`` with ``strip_constant=True``.

    Args:
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

    def __init__(self, nan='mean', null_column=None):
        super().__init__(nan=nan, null_column=null_column, strip_constant=True)
