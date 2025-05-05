import platform
import re
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from dateutil.tz import tzoffset

from rdt.errors import TransformerInputError
from rdt.transformers.datetime import (
    OptimizedTimestampEncoder,
    UnixTimestampEncoder,
)
from rdt.transformers.null import NullTransformer


class TestUnixTimestampEncoder:
    def test___init__(self):
        """Test the ``__init__`` method and the passed arguments are stored as attributes."""
        # Run
        transformer = UnixTimestampEncoder(
            missing_value_replacement='mode',
            missing_value_generation='from_column',
            datetime_format='%M-%d-%Y',
            enforce_min_max_values=True,
        )

        # Asserts
        assert transformer.missing_value_replacement == 'mode'
        assert transformer.missing_value_generation == 'from_column'
        assert transformer.datetime_format == '%M-%d-%Y'
        assert transformer.enforce_min_max_values is True

    def test___init__with_model_missing_values(self):
        """Test the ``__init__`` method and the passed arguments are stored as attributes."""
        # Run
        transformer = UnixTimestampEncoder(
            missing_value_replacement='mode',
            model_missing_values=False,
            datetime_format='%M-%d-%Y',
        )

        # Asserts
        assert transformer.missing_value_replacement == 'mode'
        assert transformer.missing_value_generation == 'random'
        assert transformer.datetime_format == '%M-%d-%Y'

    def test__convert_to_datetime(self):
        """Test the ``_convert_to_datetime`` method.

        Test to make sure the transformer converts the data to datetime
        if it is of type ``object`` and can be converted.

        Input:
            - a pandas Series of dtype object, with elements that can be
            converted to datetime.

        Output:
            - a pandas series of type datetime.
        """
        # Setup
        data = pd.Series(['2020-01-01', '2020-02-01', '2020-03-01'])
        transformer = UnixTimestampEncoder()

        # Run
        converted_data = transformer._convert_to_datetime(data)

        # Assert
        expected_data = pd.Series(pd.to_datetime(['2020-01-01', '2020-02-01', '2020-03-01']))
        pd.testing.assert_series_equal(expected_data, converted_data)

    def test__convert_to_datetime_format(self):
        """Test the ``_convert_to_datetime`` method.

        Test to make sure the transformer uses the specified format
        to convert the data to datetime.

        Setup:
            - The transformer will be initialized with a format.
        Input:
            - a pandas Series of dtype object, with elements in the specified
            format.

        Output:
            - a pandas series of type datetime.
        """
        # Setup
        data = pd.Series(['01Feb2020', '02Mar2020', '03Jan2010'])
        dt_format = '%d%b%Y'
        transformer = UnixTimestampEncoder(datetime_format=dt_format)

        # Run
        converted_data = transformer._convert_to_datetime(data)

        # Assert
        expected_data = pd.Series(pd.to_datetime(['01Feb2020', '02Mar2020', '03Jan2010']))
        pd.testing.assert_series_equal(expected_data, converted_data)

    def test__convert_to_datetime_not_convertible_raises_error(self):
        """Test the ``_convert_to_datetime`` method.

        Test to make sure a ``TypeError`` is raised if the data is of type
        ``object`` but can't be converted.

        Input:
            - a pandas Series of dtype object, with elements that can't be
            converted to datetime.

        Expected behavior:
            - a ``TypeError`` is raised.
        """
        # Setup
        data = pd.Series([
            '2020-01-01-can',
            '2020-02-01-not',
            '2020-03-01-convert',
        ])
        transformer = UnixTimestampEncoder()

        # Run
        error_message = 'Data must be of dtype datetime, or castable to datetime.'
        with pytest.raises(TypeError, match=error_message):
            transformer._convert_to_datetime(data)

    def test__convert_to_datetime_wrong_format_raises_error(self):
        """Test the ``_convert_to_datetime`` method.

        Test to make sure the transformer raises an error if the data does
        not match the specified format.

        Setup:
            - The transformer will be initialized with a format.
        Input:
            - a pandas Series of dtype object, with elements in the wrong
            format.

        Output:
            - a pandas series of type datetime.
        """
        # Setup
        data = pd.Series(['01-02-2020', '02-03-2020', '03J-01-2010'])
        dt_format = '%d%b%Y'
        transformer = UnixTimestampEncoder(datetime_format=dt_format)

        # Run
        error_message = 'Data does not match specified datetime format.'
        with pytest.raises(ValueError, match=error_message):
            transformer._convert_to_datetime(data)

    def test__transform_helper_calls_convert_to_datetime(self):
        """Test the ``_transform_helper`` method.

        Validate the helper transformer produces the correct value with ``strip_constant`` True.

        Input:
            - a pandas series of datetimes.

        Output:
            - a pandas series of the transformed datetimes.
        """
        # Setup
        data = pd.to_datetime(['2020-01-01', '2020-02-01', '2020-03-01'])
        transformer = UnixTimestampEncoder()
        transformer._convert_to_datetime = Mock()
        transformer._convert_to_datetime.return_value = data

        # Run
        transformer._transform_helper(data)

        # Assert
        transformer._convert_to_datetime.assert_called_once_with(data)

    def test__transform_helper(self):
        """Test the ``_transform_helper`` method.

        Validate the helper transformer produces the correct values.

        Input:
            - a pandas series of datetimes.

        Output:
            - a pandas series of the transformed datetimes.
        """
        # Setup
        data = pd.to_datetime(['2020-01-01', '2020-02-01', '2020-03-01'])
        transformer = UnixTimestampEncoder()

        # Run
        transformed = transformer._transform_helper(data)

        # Assert
        np.testing.assert_allclose(
            transformed,
            np.array([
                1.577837e18,
                1.580515e18,
                1.583021e18,
            ]),
            rtol=1e-5,
        )

    def test__reverse_transform_helper_nulls(self):
        """Test the ``_reverse_transform_helper`` with null values.

        Setup:
            - Mock the ``instance.null_transformer``.
            - Set the ``missing_value_replacement``.

        Input:
            - a pandas series.

        Output:
            - a pandas datetime index.

        Expected behavior:
            - The mock should call its ``reverse_transform`` method.
        """
        # Setup
        data = pd.to_datetime(['2020-01-01', '2020-02-01', '2020-03-01'])
        transformer = UnixTimestampEncoder(missing_value_replacement='mean')
        transformer.null_transformer = Mock()
        transformer.null_transformer.reverse_transform.return_value = pd.Series([1, 2, 3])

        # Run
        transformer._reverse_transform_helper(data)

        # Assert
        transformer.null_transformer.reverse_transform.assert_called_once()
        datetimes = transformer.null_transformer.reverse_transform.mock_calls[0][1][0]
        np.testing.assert_array_equal(data.to_numpy(), datetimes)

    def test__reverse_transform_helper_model_missing_values_true(self):
        """Test the ``_reverse_transform_helper`` with null values.

        Setup:
            - Mock the ``instance.null_transformer``.
            - Set the ``model_missing_values``.

        Input:
            - a pandas series.

        Output:
            - a pandas datetime index.

        Expected behavior:
            - The mock should call its ``reverse_transform`` method.
        """
        # Setup
        data = pd.to_datetime(['2020-01-01', '2020-02-01', '2020-03-01'])
        transformer = UnixTimestampEncoder(model_missing_values=True)
        transformer.null_transformer = Mock()
        transformer.null_transformer.reverse_transform.return_value = pd.Series([1, 2, 3])

        # Run
        transformer._reverse_transform_helper(data)

        # Assert
        transformer.null_transformer.reverse_transform.assert_called_once()
        datetimes = transformer.null_transformer.reverse_transform.mock_calls[0][1][0]
        np.testing.assert_array_equal(data.to_numpy(), datetimes)

    @patch('rdt.transformers.datetime.NullTransformer')
    def test__fit(self, null_transformer_mock):
        """Test the ``_fit`` method for numpy arrays.

        Validate that this method (1) sets ``self.null_transformer`` to the
        ``NullTransformer`` with the correct attributes, (2) calls the
        ``self.null_transformer``'s ``fit`` method, and (3) calls the
        ``_transform_helper`` method with the transformed data.
        """
        # Setup
        data = pd.to_datetime(['2020-01-01', '2020-02-01', '2020-03-01'])
        transformer = UnixTimestampEncoder()

        # Run
        transformer._fit(data)

        # Assert
        null_transformer_mock.assert_called_once_with('mean', 'random')
        assert null_transformer_mock.return_value.fit.call_count == 1
        np.testing.assert_allclose(
            null_transformer_mock.return_value.fit.call_args_list[0][0][0],
            np.array([1.577837e18, 1.580515e18, 1.583021e18]),
            rtol=1e-5,
        )

    def test__fit_enforce_min_max_values(self):
        """Test the ``_fit`` method when enforce_min_max_values is True.

        It should compute the min and max values of the integer conversion
        of the datetimes.
        """
        # Setup
        data = pd.to_datetime(['2020-01-01', '2020-02-01', '2020-03-01'])
        transformer = UnixTimestampEncoder(enforce_min_max_values=True)

        # Run
        transformer._fit(data)

        # Assert
        assert transformer._min_value == 1.5778368e18
        assert transformer._max_value == 1.5830208e18

    def test__fit_calls_transform_helper(self):
        """Test the ``_fit`` method.

        The ``_fit`` method should call the ``_transform_helper`` method.
        """
        # Setup
        data = pd.to_datetime(['2020-01-01', '2020-02-01', '2020-03-01'])
        transformer = UnixTimestampEncoder()
        transformer._transform_helper = Mock()
        transformer._transform_helper.return_value = pd.Series(data)

        # Run
        transformer._fit(data)

        # Assert
        transformer._transform_helper.assert_called_once()
        assert transformer.output_properties == {
            None: {'sdtype': 'float', 'next_transformer': None},
        }

    @patch('rdt.transformers.datetime._guess_datetime_format_for_array')
    def test__fit_calls_guess_datetime_format(self, mock__guess_datetime_format_for_array):
        """Test the ``_fit`` method.

        The ``_fit`` method should call the ``_transform_helper`` method.
        """
        # Setup
        data = pd.Series(['2020-02-01', '2020-03-01'])
        mock__guess_datetime_format_for_array.return_value = '%Y-%m-%d'
        transformer = UnixTimestampEncoder()

        # Run
        transformer._fit(data)

        # Assert
        np.testing.assert_array_equal(
            mock__guess_datetime_format_for_array.call_args[0][0],
            np.array(['2020-02-01', '2020-03-01']),
        )
        assert transformer.datetime_format == '%Y-%m-%d'

    def test__fit_missing_value_generation(self):
        """Test output_properties contains 'is_null' column.

        When missing_value_generation is 'from_column' the expected output is to have an extra
        column.
        """
        # Setup
        transformer = UnixTimestampEncoder(missing_value_generation='from_column')
        data = pd.Series(['2020-02-01', np.nan])

        # Run
        transformer._fit(data)

        # Assert
        assert transformer.output_properties == {
            None: {'sdtype': 'float', 'next_transformer': None},
            'is_null': {'sdtype': 'float', 'next_transformer': None},
        }

    def test__transform(self):
        """Test the ``_transform`` method for numpy arrays.

        Validate that this method calls the ``null_transformer.transform`` method.
        It should also check that the final output is correct.
        """
        # Setup
        data = pd.to_datetime(['2020-01-01', '2020-02-01', '2020-03-01'])
        transformer = UnixTimestampEncoder()
        transformer.null_transformer = Mock()

        # Run
        transformer._transform(data)

        # Assert
        assert transformer.null_transformer.transform.call_count == 1
        np.testing.assert_allclose(
            transformer.null_transformer.transform.call_args_list[0][0],
            np.array([[1.577837e18, 1.580515e18, 1.583021e18]]),
            rtol=1e-5,
        )

    def test__reverse_transform_all_none(self):
        """Test the ``_reverse_transform`` method with ``None`` values.

        Validate that the method transforms ``None`` into ``NaT``.
        """
        # Setup
        ute = UnixTimestampEncoder()
        ute.null_transformer = NullTransformer('mean')

        # Run
        output = ute._reverse_transform(pd.Series([None]))

        # Assert
        expected = pd.Series(pd.to_datetime(['NaT']))
        pd.testing.assert_series_equal(output, expected)

    def test__reverse_transform(self):
        """Test the ``_reverse_transform`` method.

        Validate that the method correctly reverse transforms.
        """
        # Setup
        ute = UnixTimestampEncoder()
        transformed = np.array([1.5778368e18, 1.5805152e18, 1.5830208e18])
        ute.null_transformer = NullTransformer('mean')

        # Run
        output = ute._reverse_transform(transformed)

        # Assert
        expected = pd.Series(pd.to_datetime(['2020-01-01', '2020-02-01', '2020-03-01']))
        pd.testing.assert_series_equal(output, expected)

    def test__reverse_transform_enforce_min_max_values(self):
        """Test the ``_reverse_transform`` with enforce_min_max_values True.

        All the values that are outside the min and max values should be clipped to the min and
        max values.
        """
        # Setup
        ute = UnixTimestampEncoder(enforce_min_max_values=True)
        transformed = np.array([
            1.5678367e18,
            1.5778368e18,
            1.5805152e18,
            1.5830208e18,
            1.5930209e18,
        ])
        ute.null_transformer = NullTransformer('mean')
        ute._min_value = 1.5778368e18
        ute._max_value = 1.5830208e18

        # Run
        output = ute._reverse_transform(transformed)

        # Assert
        expected = pd.Series(
            pd.to_datetime([
                '2020-01-01',
                '2020-01-01',
                '2020-02-01',
                '2020-03-01',
                '2020-03-01',
            ])
        )
        pd.testing.assert_series_equal(output, expected)

    def test__reverse_transform_datetime_format_dtype_is_datetime(self):
        """Test the ``_reverse_transform`` method returns the correct datetime format."""
        # Setup
        ute = UnixTimestampEncoder()
        ute.datetime_format = '%b %d, %Y'
        transformed = np.array([1.5778368e18, 1.5805152e18, 1.5830208e18])
        ute._dtype = np.dtype('<M8[ns]')
        ute.null_transformer = NullTransformer('mean')

        # Run
        output = ute._reverse_transform(transformed)

        # Assert
        expected = pd.Series(pd.to_datetime(['Jan 01, 2020', 'Feb 01, 2020', 'Mar 01, 2020']))
        pd.testing.assert_series_equal(output, expected)

    def test__reverse_transform_datetime_format(self):
        """Test the ``_reverse_transform`` method returns the correct datetime format."""
        # Setup
        ute = UnixTimestampEncoder()
        ute.datetime_format = '%b %d, %Y'
        transformed = np.array([1.5778368e18, 1.5805152e18, 1.5830208e18])
        ute._dtype = 'object'
        ute.null_transformer = NullTransformer('mean')

        # Run
        output = ute._reverse_transform(transformed)

        # Assert
        expected = pd.Series(['Jan 01, 2020', 'Feb 01, 2020', 'Mar 01, 2020'])
        pd.testing.assert_series_equal(output, expected)

    def test__reverse_transform_datetime_format_with_strftime_formats(self):
        """Test the ``_reverse_transform`` method returns the correct datetime format."""
        # Setup
        ute = UnixTimestampEncoder()
        ute.datetime_format = '%b %-d, %Y'
        transformed = np.array([1.5778368e18, 1.5805152e18, 1.5830208e18])
        ute._dtype = 'object'
        ute.null_transformer = NullTransformer('mean')

        # Run
        output = ute._reverse_transform(transformed)

        # Assert
        expected = pd.Series(['Jan 1, 2020', 'Feb 1, 2020', 'Mar 1, 2020'])
        if 'windows' not in platform.system().lower():
            pd.testing.assert_series_equal(output, expected)

    def test__reverse_transform_datetime_format_with_nans(self):
        """Test the ``_reverse_transform`` method returns the correct datetime format with nans."""
        # Setup
        ute = UnixTimestampEncoder()
        ute.datetime_format = '%b %d, %Y'
        transformed = np.array([1.5778368e18, 1.5805152e18, np.nan])
        ute._dtype = 'object'
        ute.null_transformer = NullTransformer('mean')

        # Run
        output = ute._reverse_transform(transformed)

        # Assert
        expected = pd.Series(['Jan 01, 2020', 'Feb 01, 2020', np.nan])
        pd.testing.assert_series_equal(output, expected)

    def test__reverse_transform_only_nans(self):
        """Test the ``_reverse_transform`` method returns the correct datetime format with nans."""
        # Setup
        ute = UnixTimestampEncoder()
        transformed = np.array([np.nan, np.nan, np.nan])
        ute._dtype = 'float'
        ute.null_transformer = NullTransformer('mean')

        # Run
        output = ute._reverse_transform(transformed)

        # Assert
        expected = pd.Series([np.nan, np.nan, np.nan])
        pd.testing.assert_series_equal(output, expected)

    def test__reverse_transform_missing_value_generation_from_column(self):
        """Test ``_reverse_transform`` method with `missing_value_generation` is `from_column`."""
        # Setup
        transformer = UnixTimestampEncoder(missing_value_generation='from_column')
        transformed = pd.DataFrame({
            'date': [1.5778368e18, 1.5805152e18, 1.5830208e18],
            'date.is_null': [0.1, 0.6, 0.1],
        })
        transformer._min_value = 1.5778368e18
        transformer._max_value = 1.5830208e18
        transformer.null_transformer = NullTransformer(missing_value_generation='from_column')
        transformer.null_transformer.nulls = True
        transformer.enforce_min_max_values = True

        # Run
        result = transformer._reverse_transform(transformed)

        # Assert
        expected = pd.Series(pd.to_datetime(['2020-01-01', np.nan, '2020-03-01']))
        pd.testing.assert_series_equal(result, expected)

    def test__set_fitted_parameters(self):
        """Test the ``_set_fitted_parameters`` method."""
        # Setup
        transformer = UnixTimestampEncoder()

        # Run
        transformer._set_fitted_parameters(
            'column_name',
            NullTransformer(),
            (pd.to_datetime('2022-01-02'), pd.to_datetime('2022-01-03')),
            dtype='object',
        )

        # Asserts
        assert transformer._min_value == pd.to_datetime('2022-01-02')
        assert transformer._max_value == pd.to_datetime('2022-01-03')
        assert isinstance(transformer.null_transformer, NullTransformer)
        assert transformer.columns == ['column_name']
        assert transformer.output_columns == ['column_name']
        assert transformer._dtype == 'object'

    def test__set_fitted_parameters_no_min_max(self):
        """Test ``_set_fitted_parameters`` sets the required parameters for transformer."""
        # Setup
        transformer = UnixTimestampEncoder(enforce_min_max_values=True)
        error_msg = re.escape('Must provide min and max values for this transformer.')
        # Run
        with pytest.raises(TransformerInputError, match=error_msg):
            transformer._set_fitted_parameters(
                'column_name',
                null_transformer=NullTransformer(),
                dtype='object',
            )

    def test__set_fitted_parameters_from_column(self):
        """Test ``_set_fitted_parameters`` sets the required parameters for transformer."""
        # Setup
        transformer = UnixTimestampEncoder()
        null_transformer = NullTransformer(missing_value_generation='from_column')

        # Run
        transformer._set_fitted_parameters(
            'column_name',
            null_transformer=null_transformer,
        )

        # Assert
        assert transformer.columns == ['column_name']
        assert transformer.output_columns == ['column_name', 'column_name.is_null']
        assert transformer.null_transformer == null_transformer
        assert transformer._min_value is None
        assert transformer._max_value is None
        assert transformer._dtype == 'object'

    def test__convert_data_to_pandas_datetime_without_utc(self):
        """Test _convert_data_to_pandas_datetime without UTC when no %z in format."""
        # Setup
        transformer = UnixTimestampEncoder()
        transformer.datetime_format = '%Y-%m-%d %H:%M:%S'
        data = pd.Series([1.706668341234567e18])

        # Run
        result = transformer._convert_data_to_pandas_datetime(data)

        # Assert
        assert result.dt.tz is None

    def test__convert_data_to_pandas_datetime_with_multiple_timezones(self):
        """Test `_convert_data_to_pandas_datetime` with multiple timezones."""
        # Setup
        transformer = UnixTimestampEncoder()
        transformer.datetime_format = '%Y-%m-%d %H:%M:%S%Z'
        transformer._has_multiple_timezones = True
        data = pd.Series([1.706668341234567e18])

        # Run
        result = transformer._convert_data_to_pandas_datetime(data)

        # Assert
        assert str(result.dt.tz) == 'UTC'

    def test__convert_data_to_pandas_datetime_with_timezone(self):
        """Test `_convert_data_to_pandas_datetime` with a single timezone."""
        # Setup
        transformer = UnixTimestampEncoder()
        transformer.datetime_format = '%Y-%m-%d %H:%M:%S%z'
        transformer._timezone_offset = tzoffset(name='EDT', offset=-5 * 3600)
        transformer._has_multiple_timezones = False
        data = pd.Series([1.706668341234567e18])

        # Run
        result = transformer._convert_data_to_pandas_datetime(data)

        # Assert
        assert str(result.dt.tz) == "tzoffset('EDT', -18000)"
        assert result.dt.strftime('%z').iloc[0] == '-0500'

    def test__handle_datetime_formatting_with_format_strftime_like(self):
        """Test `_handle_datetime_formatting` applies formatting when `datetime_format` is set."""
        # Setup
        transformer = UnixTimestampEncoder()
        transformer.datetime_format = '%b %d, %Y'
        transformer._dtype = 'object'
        data = pd.to_datetime(['2025-04-30'])

        # Run
        result = transformer._handle_datetime_formatting(pd.Series(data))

        # Assert
        expected = pd.Series(['Apr 30, 2025'])
        pd.testing.assert_series_equal(result, expected)

    def test__handle_datetime_formatting_with_numeric_dtype(self):
        """Test `_handle_datetime_formatting` coerces to numeric when `_dtype` is numeric."""
        # Setup
        transformer = UnixTimestampEncoder()
        transformer.datetime_format = None
        transformer._dtype = 'float64'
        data = pd.to_datetime(['2025-04-30'])

        # Run
        result = transformer._handle_datetime_formatting(pd.Series(data))

        # Assert
        assert result.dtype == 'float64'

    def test__format_datetime_strftime_only(self):
        """Test `_format_datetime` returns formatted strings when dtype is object."""
        # Setup
        transformer = UnixTimestampEncoder()
        transformer.datetime_format = '%Y-%m-%d'
        transformer._dtype = 'object'
        data = pd.to_datetime(['2025-01-01'])

        # Run
        result = transformer._format_datetime(pd.Series(data))

        # Assert
        expected = pd.Series(['2025-01-01'], dtype='object')
        pd.testing.assert_series_equal(result, expected)

    def test__reverse_transform_calls_chain_of_steps(self):
        """Test that `_reverse_transform` calls a series methods in order."""
        # Setup
        instance = Mock()
        data_object = object()
        instance.enforce_min_max_values = False

        # Run
        result = UnixTimestampEncoder._reverse_transform(instance, data_object)

        # Assert
        instance._reverse_transform_helper.assert_called_once_with(data_object)
        instance._convert_data_to_pandas_datetime.assert_called_once_with(
            instance._reverse_transform_helper.return_value
        )
        instance._handle_datetime_formatting.assert_called_once_with(
            instance._convert_data_to_pandas_datetime.return_value
        )
        assert result == instance._handle_datetime_formatting.return_value

    def test__convert_to_datetime_has_multiple_timezones(self):
        """Test the `_convert_to_datetime` method returns the input data when no need to convert."""
        # Setup
        instance = Mock()
        data_object = object()
        instance._needs_datetime_conversion.return_value = False

        # Run
        result = UnixTimestampEncoder._convert_to_datetime(instance, data_object)

        # Assert
        assert data_object == result

    def test__warn_if_mixed_timezones(self):
        """Test that a warning is being shown when the data contains mixed timezones."""
        # Setup
        instance = Mock()
        instance._has_multiple_timezones = True

        # Run
        warning_msg = (
            'Mixed timezones are not supported in SDV Community. Data will be converted to UTC.'
        )
        with pytest.warns(UserWarning, match=warning_msg):
            UnixTimestampEncoder._warn_if_mixed_timezones(instance)

    def test__learn_timezone_offest_sets_timezone_offset(self):
        """Test that `_learn_timezone_offest` sets `_timezone_offset` correctly.

        Setup:
            - Instance with `datetime_format` including `%z` and `_has_multiple_timezones=False`.
            - Input data contains one or more valid datetime strings with timezone offset.
        Input:
            - A list of datetime strings with one valid `%z`-formatted entry.
        Output:
            - `_timezone_offset` is set to the tzinfo of the first valid datetime.
        """
        # Setup
        instance = UnixTimestampEncoder()
        instance.datetime_format = '%Y-%m-%d %H:%M:%S%z'
        instance._has_multiple_timezones = False
        data = [None, 'invalid-date', '2025-04-01 14:00:00+0300', '2025-04-02 14:00:00+0300']

        # Run
        UnixTimestampEncoder._learn_timezone_offest(instance, data)

        # Assert
        assert instance._timezone_offset.utcoffset(None).total_seconds() == 10800

    def test__learn_timezone_offest_with_no_valid_dates(self):
        """Test that `_learn_timezone_offest` with no valid data."""
        # Setup
        instance = UnixTimestampEncoder()
        instance.datetime_format = '%Y-%m-%d %H:%M:%S%z'
        instance._has_multiple_timezones = False
        data = [None, 'not-a-date', '---']

        # Run
        UnixTimestampEncoder._learn_timezone_offest(instance, data)

        # Assert
        assert instance._timezone_offset is None

    @patch('rdt.transformers.datetime.data_has_multiple_timezones')
    def test__learn_has_multiple_timezones_with_no_valid_timezones(self, mock_data_has_multiple_tz):
        """Test that `_learn_has_multiple_timezones` handles data with no valid timezones."""
        # Setup
        instance = UnixTimestampEncoder()
        instance.datetime_format = '%Y-%m-%d %H:%M:%S%z'
        data = pd.Series([None, 'not-a-date', '---'])
        mock_data_has_multiple_tz.return_value = False

        # Run
        UnixTimestampEncoder._learn_has_multiple_timezones(instance, data)

        # Assert
        assert instance._has_multiple_timezones is False
        mock_data_has_multiple_tz.assert_called_once_with(data)

    @patch('rdt.transformers.datetime.is_numeric_dtype')
    def test__needs_datetime_conversion_with_non_numeric_data(self, mock_pd_is_numeric_dtype):
        """Test that `_needs_datetime_conversion` returns True for non-numeric data."""
        # Setup
        instance = UnixTimestampEncoder()
        instance.datetime_format = '%Y-%m-%d'
        data = pd.Series(['2023-10-15', '2023-10-16'])

        # Run
        result = instance._needs_datetime_conversion(data)

        # Assert
        assert result is True
        mock_pd_is_numeric_dtype.assert_not_called()

    @patch('rdt.transformers.datetime.is_numeric_dtype', return_value=True)
    def test__needs_datetime_conversion_with_numeric_data(self, mock_pd_is_numeric_dtype):
        """Test that `_needs_datetime_conversion` returns False for numeric data with no format."""
        # Setup
        instance = UnixTimestampEncoder()
        instance.datetime_format = None
        data = pd.Series([1234567890, 1234567891])

        # Run
        result = instance._needs_datetime_conversion(data)

        # Assert
        assert result is False
        mock_pd_is_numeric_dtype.assert_called_once_with(data)

    @patch('rdt.transformers.datetime.pd.to_datetime')
    def test__to_datetime_with_no_timezones(self, mock_pd_to_datetime):
        """Test that `_to_datetime` converts data without timezones."""
        # Setup
        instance = UnixTimestampEncoder()
        instance._has_multiple_timezones = False
        instance.datetime_format = '%Y-%m-%d'
        data = pd.Series(['2023-10-15', '2023-10-16'])
        mock_pd_to_datetime.return_value = pd.Series([
            pd.Timestamp('2023-10-15'),
            pd.Timestamp('2023-10-16'),
        ])

        # Run
        result = instance._to_datetime(data)

        # Assert
        expected = pd.Series([pd.Timestamp('2023-10-15'), pd.Timestamp('2023-10-16')])
        pd.testing.assert_series_equal(result, expected)
        mock_pd_to_datetime.assert_called_once_with(data, format='%Y-%m-%d', utc=False)

    @patch('rdt.transformers.datetime.pd.to_datetime')
    def test__to_datetime_with_multiple_timezones(self, mock_pd_to_datetime):
        """Test that `_to_datetime` converts data to UTC with multiple timezones."""
        # Setup
        instance = UnixTimestampEncoder()
        instance._has_multiple_timezones = True
        instance.datetime_format = '%Y-%m-%d %H:%M:%S%z'
        data = pd.Series(['2023-10-15 14:30:00+0200', '2023-10-15 14:30:00-0500'])
        mock_pd_to_datetime.return_value = pd.Series([
            pd.Timestamp('2023-10-15 12:30:00+0000', tz='UTC'),
            pd.Timestamp('2023-10-15 19:30:00+0000', tz='UTC'),
        ])

        # Run
        result = instance._to_datetime(data)

        # Assert
        expected = pd.Series([
            pd.Timestamp('2023-10-15 12:30:00+0000', tz='UTC'),
            pd.Timestamp('2023-10-15 19:30:00+0000', tz='UTC'),
        ])
        pd.testing.assert_series_equal(result, expected)
        mock_pd_to_datetime.assert_called_once_with(data, format='%Y-%m-%d %H:%M:%S%z', utc=True)

    def test__get_pandas_datetime_format_with_format_windows(self):
        """Test that `_get_pandas_datetime_format` converts format to pandas-compatible."""
        # Setup
        instance = UnixTimestampEncoder()
        instance.datetime_format = '%Y-%#m-%-d'

        # Run
        result = instance._get_pandas_datetime_format()

        # Assert
        assert result == '%Y-%m-%d'

    def test__get_pandas_datetime_format_with_format_unix(self):
        """Test that `_get_pandas_datetime_format` converts format to pandas-compatible."""
        # Setup
        instance = UnixTimestampEncoder()
        instance.datetime_format = '%Y-%-m-%-d'

        # Run
        result = instance._get_pandas_datetime_format()

        # Assert
        assert result == '%Y-%m-%d'

    def test__get_pandas_datetime_format_with_no_format(self):
        """Test that `_get_pandas_datetime_format` returns None when no format is set."""
        # Setup
        instance = UnixTimestampEncoder()
        instance.datetime_format = None

        # Run
        result = instance._get_pandas_datetime_format()

        # Assert
        assert result is None


class TestOptimizedTimestampEncoder:
    def test___init__(self):
        """Test the ``__init__`` method."""
        # Run
        transformer = OptimizedTimestampEncoder(
            missing_value_replacement='mode',
            missing_value_generation='from_column',
            datetime_format='%M-%d-%Y',
            enforce_min_max_values=True,
        )

        # Asserts
        assert transformer.enforce_min_max_values is True
        assert transformer.missing_value_replacement == 'mode'
        assert transformer.missing_value_generation == 'from_column'
        assert transformer.datetime_format == '%M-%d-%Y'
        assert transformer.divider is None
        assert transformer.null_transformer is None

    def test__find_divider(self):
        """Test the ``_find_divider`` method.

        Find the greatest common denominator out of these values: [10] * 9 + [60, 60, 24],
        where each consecutive value in the list is multiplied by the previous one
        (so 10, 100, 1000, etc).
        """
        # Setup
        data = np.array([100, 7919])
        transformer = OptimizedTimestampEncoder()

        # Run
        transformer._find_divider(data)

        # Assert
        assert transformer.divider == 1

    def test__transform_helper(self):
        """Test the ``_transform_helper`` method.

        Validate the helper method produces the values stripped to the smallest
        non-zero time unit.
        """
        # Setup
        data = pd.to_datetime(['2020-01-01', '2020-02-01', '2020-03-01'])
        transformer = OptimizedTimestampEncoder()

        # Run
        transformed = transformer._transform_helper(data)

        # Assert
        np.testing.assert_allclose(
            transformed,
            np.array([
                18262.0,
                18293.0,
                18322.0,
            ]),
        )

    def test__reverse_transform_helper(self):
        """Test the ``_reverse_transform_helper`` method.

        Validate the helper produces the values multiplied by the
        smallest non-zero time unit.
        """
        # Setup
        data = pd.Series([18262.0, 18293.0, 18322.0])
        transformer = OptimizedTimestampEncoder()
        transformer.divider = 1000
        transformer.null_transformer = Mock()
        transformer.null_transformer.reverse_transform.side_effect = lambda x: x

        # Run
        multiplied = transformer._reverse_transform_helper(data)

        # Assert
        expected = np.array([18262000, 18293000, 18322000])
        np.testing.assert_allclose(multiplied, expected)
