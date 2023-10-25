import platform
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from rdt.transformers.datetime import OptimizedTimestampEncoder, UnixTimestampEncoder
from rdt.transformers.null import NullTransformer


class TestUnixTimestampEncoder:

    def test___init__(self):
        """Test the ``__init__`` method and the passed arguments are stored as attributes."""
        # Run
        transformer = UnixTimestampEncoder(
            missing_value_replacement='mode',
            missing_value_generation='from_column',
            datetime_format='%M-%d-%Y'
        )

        # Asserts
        assert transformer.missing_value_replacement == 'mode'
        assert transformer.missing_value_generation == 'from_column'
        assert transformer.datetime_format == '%M-%d-%Y'

    def test___init__with_model_missing_values(self):
        """Test the ``__init__`` method and the passed arguments are stored as attributes."""
        # Run
        transformer = UnixTimestampEncoder(
            missing_value_replacement='mode',
            model_missing_values=False,
            datetime_format='%M-%d-%Y'
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
        data = pd.Series(['2020-01-01-can', '2020-02-01-not', '2020-03-01-convert'])
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
        np.testing.assert_allclose(transformed, np.array([
            1.577837e+18, 1.580515e+18, 1.583021e+18,
        ]), rtol=1e-5)

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
            np.array([1.577837e+18, 1.580515e+18, 1.583021e+18]), rtol=1e-5
        )

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
            np.array(['2020-02-01', '2020-03-01'])
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
            np.array([[1.577837e+18, 1.580515e+18, 1.583021e+18]]), rtol=1e-5
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
        transformed = np.array([1.5778368e+18, 1.5805152e+18, 1.5830208e+18])
        ute.null_transformer = NullTransformer('mean')

        # Run
        output = ute._reverse_transform(transformed)

        # Assert
        expected = pd.Series(pd.to_datetime(['2020-01-01', '2020-02-01', '2020-03-01']))
        pd.testing.assert_series_equal(output, expected)

    def test__reverse_transform_datetime_format_dtype_is_datetime(self):
        """Test the ``_reverse_transform`` method returns the correct datetime format."""
        # Setup
        ute = UnixTimestampEncoder()
        ute.datetime_format = '%b %d, %Y'
        transformed = np.array([1.5778368e+18, 1.5805152e+18, 1.5830208e+18])
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
        transformed = np.array([1.5778368e+18, 1.5805152e+18, 1.5830208e+18])
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
        transformed = np.array([1.5778368e+18, 1.5805152e+18, 1.5830208e+18])
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
        transformed = np.array([1.5778368e+18, 1.5805152e+18, np.nan])
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


class TestOptimizedTimestampEncoder:

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
        np.testing.assert_allclose(transformed, np.array([
            18262., 18293., 18322.,
        ]))

    def test__reverse_transform_helper(self):
        """Test the ``_reverse_transform_helper`` method.

        Validate the helper produces the values multiplied by the
        smallest non-zero time unit.
        """
        # Setup
        data = pd.Series([18262., 18293., 18322.])
        transformer = OptimizedTimestampEncoder()
        transformer.divider = 1000
        transformer.null_transformer = Mock()
        transformer.null_transformer.reverse_transform.side_effect = lambda x: x

        # Run
        multiplied = transformer._reverse_transform_helper(data)

        # Assert
        expected = np.array([18262000, 18293000, 18322000])
        np.testing.assert_allclose(multiplied, expected)
