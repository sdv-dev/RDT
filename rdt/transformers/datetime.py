"""Transformer for datetime data."""

import warnings

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_dtype, is_numeric_dtype
from pandas.core.tools.datetimes import _guess_datetime_format_for_array

from rdt.errors import TransformerInputError
from rdt.transformers.base import BaseTransformer
from rdt.transformers.null import NullTransformer
from rdt.transformers.utils import _safe_parse_datetime, data_has_multiple_timezones


class UnixTimestampEncoder(BaseTransformer):
    """Transformer for datetime data.

    This transformer replaces datetime values with an integer timestamp
    transformed to float.

    Null values are replaced using a ``NullTransformer``.

    Args:
        missing_value_replacement (object):
            Indicate what to replace the null values with. If the strings ``'mean'`` or ``'mode'``
            are given, replace them with the corresponding aggregation, if ``'random'``, use
            random values from the dataset to fill the nan values.
            Defaults to ``mean``.
        model_missing_values (bool):
            **DEPRECATED** Whether to create a new column to indicate which values were null or
            not. The column will be created only if there are null values. If ``True``, create
            the new column if there are null values. If ``False``, do not create the new column
            even if there are null values. Defaults to ``False``.
        datetime_format (str):
            The strftime to use for parsing time. For more information, see
            https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior.
        missing_value_generation (str or None):
            The way missing values are being handled. There are three strategies:

                * ``random``: Randomly generates missing values based on the percentage of
                  missing values.
                * ``from_column``: Creates a binary column that describes whether the original
                  value was missing. Then use it to recreate missing values.
                * ``None``: Do nothing with the missing values on the reverse transform. Simply
                  pass whatever data we get through.
        enforce_min_max_values (bool):
            Whether or not to clip the data returned by ``reverse_transform`` to the min and
            max values seen during ``fit``. Defaults to ``False``.
    """

    INPUT_SDTYPE = 'datetime'
    null_transformer = None
    _min_value = None
    _max_value = None

    def __init__(
        self,
        missing_value_replacement='mean',
        model_missing_values=None,
        datetime_format=None,
        missing_value_generation='random',
        enforce_min_max_values=False,
    ):
        super().__init__()
        self.missing_value_replacement = missing_value_replacement
        self._set_missing_value_generation(missing_value_generation)
        self.enforce_min_max_values = enforce_min_max_values
        if model_missing_values is not None:
            self._set_model_missing_values(model_missing_values)

        self.datetime_format = datetime_format
        self._dtype = None
        self._has_multiple_timezones = False
        self._timezone_offset = None

    def _warn_if_mixed_timezones(self):
        if self._has_multiple_timezones:
            warnings.warn(
                'Mixed timezones are not supported in SDV Community. Data will be converted to UTC.'
            )

    def _raise_appropriate_conversion_error(self, error):
        message = str(error)
        if 'Unknown string' in message or 'Unknown datetime string' in message:
            raise TypeError('Data must be of dtype datetime, or castable to datetime.') from None

        raise ValueError('Data does not match specified datetime format.') from None

    def _learn_timezone_offset(self, data):
        """Extracts and stores the timezone offset from the first valid datetime in the data."""
        has_timezone = self.datetime_format and '%z' in self.datetime_format.lower()
        if has_timezone and not self._has_multiple_timezones:
            for val in data:
                if pd.notna(val):
                    try:
                        dt = _safe_parse_datetime(
                            str(val), warn=True, datetime_format=self.datetime_format
                        )
                        self._timezone_offset = dt.tzinfo
                        break

                    except (ValueError, TypeError, AttributeError):
                        self._timezone_offset = None

    def _learn_has_multiple_timezones(self, data):
        """Determines if the data contains multiple timezones and stores the result."""
        if self.datetime_format and '%z' not in self.datetime_format.lower():
            return

        if not isinstance(data, pd.Series):
            data = data.to_series()

        self._has_multiple_timezones = data_has_multiple_timezones(
            data, datetime_format=self.datetime_format
        )

    def _needs_datetime_conversion(self, data):
        """Determines if the data requires datetime conversion."""
        return self.datetime_format is not None or not is_numeric_dtype(data)

    def _get_pandas_datetime_format(self):
        """Converts the instance's datetime format to a pandas-compatible format."""
        if self.datetime_format:
            return self.datetime_format.replace('%#', '%').replace('%-', '%')

        return None

    def _to_datetime(self, data):
        """Converts data to datetime using the instance's datetime format."""
        return pd.to_datetime(
            data,
            format=self._get_pandas_datetime_format(),
            utc=getattr(self, '_has_multiple_timezones', False),  # Backward compatibility
        )

    def _convert_to_datetime(self, data):
        if self._needs_datetime_conversion(data):
            try:
                return self._to_datetime(data)
            except ValueError as error:
                self._raise_appropriate_conversion_error(error)

        return data

    def _transform_helper(self, datetimes):
        """Transform datetime values to integer."""
        datetimes = self._convert_to_datetime(datetimes)
        nulls = datetimes.isna()
        integers = pd.to_numeric(datetimes, errors='coerce').to_numpy().astype(np.float64)
        integers[nulls] = np.nan
        transformed = pd.Series(integers)

        return transformed

    def _fit(self, data):
        """Fit the transformer to the data.

        Args:
            data (pandas.Series):
                Data to fit the transformer to.
        """
        self._dtype = data.dtype
        if self.datetime_format is None:
            datetime_array = data[data.notna()].astype(str).to_numpy()
            self.datetime_format = _guess_datetime_format_for_array(datetime_array)

        self._learn_has_multiple_timezones(data)
        transformed = self._transform_helper(data)
        self._learn_timezone_offset(data)
        self._warn_if_mixed_timezones()

        if self.enforce_min_max_values:
            self._min_value = transformed.min()
            self._max_value = transformed.max()

        self.null_transformer = NullTransformer(
            self.missing_value_replacement, self.missing_value_generation
        )
        self.null_transformer.fit(transformed)
        if self.null_transformer.models_missing_values():
            self.output_properties['is_null'] = {
                'sdtype': 'float',
                'next_transformer': None,
            }

    def _set_fitted_parameters(
        self, column_name, null_transformer, min_max_values=None, dtype='object'
    ):
        """Manually set the parameters on the transformer to get it into a fitted state.

        Args:
            column_name (str):
                The name of the column for this transformer.
            null_transformer (NullTransformer):
                A fitted null transformer instance that can be used to generate
                null values for the column.
            min_max_values (tuple or None):
                None or a tuple containing the (min, max) values for the transformer.
                Should be used to set self._min_value and self._max_value and must be
                provided if self.enforce_min_max_values is True.
                Defaults to None.
            dtype (str, optional):
                The dtype to convert the reverse transformed data back to. Defaults to 'object'.
        """
        self.reset_randomization()
        self.columns = [column_name]
        self.output_columns = [column_name]
        self._dtype = dtype

        if self.enforce_min_max_values and not min_max_values:
            raise TransformerInputError('Must provide min and max values for this transformer.')

        if min_max_values:
            self._min_value = min_max_values[0]
            self._max_value = min_max_values[1]

        self.null_transformer = null_transformer
        if self.null_transformer.models_missing_values():
            self.output_columns.append(column_name + '.is_null')

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

    def _convert_data_to_pandas_datetime(self, data):
        """Converts data to pandas datetime, applying UTC or specific timezone offset if set."""
        if hasattr(self, '_has_multiple_timezones') and self._has_multiple_timezones:
            return pd.to_datetime(data, utc=True)

        elif hasattr(self, '_timezone_offset') and self._timezone_offset is not None:
            datime_data = pd.to_datetime(data, utc=True)
            return datime_data.dt.tz_convert(self._timezone_offset)

        return pd.to_datetime(data)

    def _handle_datetime_formatting(self, datetime_data):
        """Formats datetime data based on datetime format or converts to numeric if needed."""
        if self.datetime_format:
            return self._format_datetime(datetime_data)

        elif is_numeric_dtype(self._dtype):
            datetime_data = pd.to_numeric(datetime_data.astype('object'), errors='coerce')
            return datetime_data.astype(self._dtype)

        return datetime_data

    def _format_datetime(self, datetime_data):
        """Formats datetime data to string or datetime using learned datatime format and dtype."""
        if is_datetime64_dtype(self._dtype) and '.%f' not in self.datetime_format:
            return pd.to_datetime(
                datetime_data.dt.strftime(self.datetime_format), format=self.datetime_format
            )

        return datetime_data.dt.strftime(self.datetime_format).astype(self._dtype)

    def _reverse_transform_helper(self, data):
        """Transform integer values back into datetimes."""
        if not isinstance(data, np.ndarray):
            data = data.to_numpy()

        data = self.null_transformer.reverse_transform(data)
        data = np.round(data.astype(np.float64))
        return data

    def _reverse_transform(self, data):
        """Convert float values back to datetimes.

        Args:
            data (pandas.Series or numpy.ndarray):
                Data to transform.

        Returns:
            pandas.Series
        """
        data = self._reverse_transform_helper(data)
        if self.enforce_min_max_values:
            data = data.clip(self._min_value, self._max_value)

        datetime_data = self._convert_data_to_pandas_datetime(data)
        datetime_data = self._handle_datetime_formatting(datetime_data)
        return datetime_data


class OptimizedTimestampEncoder(UnixTimestampEncoder):
    """Optimized transformer for datetime data.

    This transformer replaces datetime values with an integer timestamp transformed to float.
    It optimizes the output values by finding the smallest time unit that is not zero on
    the training datetimes and dividing the generated numerical values by the value of the next
    smallest time unit. This, apart from reducing the orders of magnitude of the transformed
    values, ensures that reverted values always are zero on the lower time units.

    Null values are replaced using a ``NullTransformer``.

    This class behaves exactly as the ``UnixTimestampEncoder`` except with the optimization.

    Args:
        missing_value_replacement (object):
            Indicate what to replace the null values with. If the strings ``'mean'`` or ``'mode'``
            are given, replace them with the corresponding aggregation, if ``'random'``, use
            random values from the dataset to fill the nan values.
            Defaults to ``mean``.
        model_missing_values (bool):
            **DEPRECATED** Whether to create a new column to indicate which values were null or
            not. The column will be created only if there are null values. If ``True``, create
            the new column if there are null values. If ``False``, do not create the new column
            even if there are null values. Defaults to ``False``.
        datetime_format (str):
            The strftime to use for parsing time. For more information, see
            https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior.
        missing_value_generation (str or None):
            The way missing values are being handled. There are three strategies:

                * ``random``: Randomly generates missing values based on the percentage of
                  missing values.
                * ``from_column``: Creates a binary column that describes whether the original
                  value was missing. Then use it to recreate missing values.
                * ``None``: Do nothing with the missing values on the reverse transform. Simply
                  pass whatever data we get through.
        enforce_min_max_values (bool):
            Whether or not to clip the data returned by ``reverse_transform`` to the min and
            max values seen during ``fit``. Defaults to ``False``.
    """

    divider = None

    def _find_divider(self, transformed):
        self.divider = 1
        multipliers = [10] * 9 + [60, 60, 24]
        for multiplier in multipliers:
            candidate = self.divider * multiplier
            if (transformed % candidate).any():
                break

            self.divider = candidate

    def _transform_helper(self, data):
        """Transform datetime values to integer."""
        data = super()._transform_helper(data)
        self._find_divider(data)
        return data // self.divider

    def _reverse_transform_helper(self, data):
        """Transform integer values back into datetimes."""
        data = super()._reverse_transform_helper(data)
        return data * self.divider
