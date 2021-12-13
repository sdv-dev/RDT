from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from rdt.transformers.datetime import DatetimeRoundedTransformer, DatetimeTransformer
from rdt.transformers.null import NullTransformer


class TestDatetimeTransformer:

    def test___init__(self):
        """Test the ``__init__`` method.

        Validate the passed arguments are stored as attributes.

        Setup:
            - initialize a ``DatetimeTransformer`` with values for each parameter.

        Side effect:
            - the ``nan`` attribute has been assigned as ``'mode'``.
            - the ``null_column`` attribute has been assigned as True.
            - the ``strip_constant`` attribute has been assigned as True.
        """
        # Setup
        transformer = DatetimeTransformer(
            nan='mode',
            null_column=True,
            strip_constant=True
        )

        # Asserts
        assert transformer.nan == 'mode'
        assert transformer.null_column is True
        assert transformer.strip_constant is True

    def test_is_composition_identity_null_transformer_true(self):
        """Test the ``is_composition_identity`` method with a ``null_transformer``.

        When the attribute ``null_transformer`` is not None and a null column is not created,
        this method should simply return False.

        Setup:
            - initialize a ``DatetimeTransformer`` transformer which sets
            ``self.null_transformer`` to a ``NullTransformer``.

        Output:
            - False.
        """
        # Setup
        transformer = DatetimeTransformer()
        transformer.null_transformer = NullTransformer(fill_value='fill')

        # Run
        output = transformer.is_composition_identity()

        # Assert
        assert output is False

    def test_is_composition_identity_null_transformer_false(self):
        """Test the ``is_composition_identity`` method without a ``null_transformer``.

        When the attribute ``null_transformer`` is None, this method should return
        the value stored in the ``COMPOSITION_IS_IDENTITY`` attribute.

        Setup:
            - initialize a ``DatetimeTransformer`` transformer which sets
            ``self.null_transformer`` to None.

        Output:
            - the value stored in ``self.COMPOSITION_IS_IDENTITY``.
        """
        # Setup
        transformer = DatetimeTransformer()
        transformer.null_transformer = None

        # Run
        output = transformer.is_composition_identity()

        # Assert
        assert output is True

    def test_get_output_types(self):
        """Test the ``get_output_types`` method when a null column is created.

        When a null column is created, this method should apply the ``_add_prefix``
        method to the following dictionary of output types:

        output_types = {
            'value': 'float',
            'is_null': 'float'
        }

        Setup:
            - initialize a ``DatetimeTransformer`` transformer which:
                - sets ``self.null_transformer`` to a ``NullTransformer`` where
                ``self._null_column`` is True.
                - sets ``self.column_prefix`` to a column name.

        Output:
            - the ``output_types`` dictionary, but with the ``self.column_prefix``
            added to the beginning of the keys.
        """
        # Setup
        transformer = DatetimeTransformer()
        transformer.null_transformer = NullTransformer(fill_value='fill')
        transformer.null_transformer._null_column = True
        transformer.column_prefix = 'a#b'

        # Run
        output = transformer.get_output_types()

        # Assert
        expected = {
            'a#b.value': 'float',
            'a#b.is_null': 'float'
        }
        assert output == expected

    def test__find_divider(self):
        """Test the ``_find_divider`` method.

        Find the greatest common denominator out of these values: [10] * 9 + [60, 60, 24],
        where each consecutive value in the list is multiplied by the previous one
        (so 10, 100, 1000, etc).

        Input:
            - a numpy array.

        Side effect:
            - sets ``self.divider`` to the correct divider.
        """
        # Setup
        data = np.array([100, 7919])
        transformer = DatetimeTransformer()

        # Run
        transformer._find_divider(data)

        # Assert
        assert transformer.divider == 1

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
        transformer = DatetimeTransformer()

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
        transformer = DatetimeTransformer(datetime_format=dt_format)

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
        transformer = DatetimeTransformer()

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
        transformer = DatetimeTransformer(datetime_format=dt_format)

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
        transformer = DatetimeTransformer()
        transformer._convert_to_datetime = Mock()
        transformer._convert_to_datetime.return_value = data

        # Run
        transformer._transform_helper(data)

        # Assert
        transformer._convert_to_datetime.assert_called_once_with(data)

    def test__transform_helper_strip_constant_true(self):
        """Test the ``_transform_helper`` method.

        Validate the helper transformer produces the correct value with ``strip_constant`` True.

        Input:
            - a pandas series of datetimes.

        Output:
            - a pandas series of the transformed datetimes.
        """
        # Setup
        data = pd.to_datetime(['2020-01-01', '2020-02-01', '2020-03-01'])
        transformer = DatetimeTransformer(strip_constant=True)

        # Run
        transformed = transformer._transform_helper(data)

        # Assert
        np.testing.assert_allclose(transformed, np.array([
            18262., 18293., 18322.,
        ]))

    def test__transform_helper_strip_constant_false(self):
        """Test the ``_transform_helper`` method.

        Validate the helper transformer produces the correct value with ``strip_constant`` False.

        Input:
            - a pandas series of datetimes.

        Output:
            - a pandas series of the transformed datetimes.
        """
        # Setup
        data = pd.to_datetime(['2020-01-01', '2020-02-01', '2020-03-01'])
        transformer = DatetimeTransformer(strip_constant=False)

        # Run
        transformed = transformer._transform_helper(data)

        # Assert
        np.testing.assert_allclose(transformed, np.array([
            1.577837e+18, 1.580515e+18, 1.583021e+18,
        ]), rtol=1e-5)

    @patch('rdt.transformers.datetime.NullTransformer')
    def test__fit(self, null_transformer_mock):
        """Test the ``_fit`` method for numpy arrays.

        Validate that this method (1) sets ``self.null_transformer`` to the
        ``NullTransformer`` with the correct attributes, (2) calls the
        ``self.null_transformer``'s ``fit`` method, and (3) calls the
        ``_transform_helper`` method with the transformed data.

        Input:
            - a pandas Series.

        Side effects:
            - sets ``self.null_transformer`` to the ``NullTransformer`` with
            the correct attributes.
            - calls the ``self.null_transformer``'s ``fit`` method.
            - calls the ``_transform_helper`` method with the transformed data.
        """
        # Setup
        data = pd.to_datetime(['2020-01-01', '2020-02-01', '2020-03-01'])
        transformer = DatetimeTransformer()

        # Run
        transformer._fit(data)

        # Assert
        null_transformer_mock.assert_called_once_with('mean', None, copy=True)
        assert null_transformer_mock.return_value.fit.call_count == 1
        np.testing.assert_allclose(
            null_transformer_mock.return_value.fit.call_args_list[0][0],
            np.array([[1.577837e+18, 1.580515e+18, 1.583021e+18]]), rtol=1e-5
        )

    def test__transform(self):
        """Test the ``_transform`` method for numpy arrays.

        Validate that this method calls the ``null_transformer.transform`` method.
        It should also check that the final output is correct.

        Setup:
            - mock behavior of ``_fit``.

        Input:
            - a pandas Series.

        Output:
            - a numpy array containing the transformed data.

        Side effect:
            - calls the ``null_transformer.transform`` method with the transformed data.
        """
        # Setup
        data = pd.to_datetime(['2020-01-01', '2020-02-01', '2020-03-01'])
        transformer = DatetimeTransformer()
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
        dt = pd.to_datetime(['2020-01-01'])
        dtt = DatetimeTransformer(strip_constant=True)
        dtt._fit(dt)

        output = dtt._reverse_transform(pd.Series([None]))

        expected = pd.Series(pd.to_datetime(['NaT']))
        pd.testing.assert_series_equal(output, expected)

    def test__reverse_transform_2d_ndarray(self):
        """Test the ``_reverese_transform`` method for 2d arrays.

        Validate that the method correctly reverse transforms 2d arrays.

        Input:
            - a numpy 2d array.

        Output:
            - a pandas Series of datetimes.
        """
        # Setup
        dt = pd.to_datetime(['2020-01-01', '2020-02-01', '2020-03-01'])
        dtt = DatetimeTransformer(nan=None, strip_constant=True)
        dtt._fit(dt)
        transformed = np.array([[18262.], [18293.], [18322.]])

        # Run
        output = dtt._reverse_transform(transformed)

        # Assert
        expected = pd.to_datetime(['2020-01-01', '2020-02-01', '2020-03-01'])
        pd.testing.assert_series_equal(output.to_series(), expected.to_series())


class TestDatetimeRoundedTransformer:

    def test___init___strip_is_true(self):
        """Test that by default the ``strip_constant`` is set to True."""
        dtrt = DatetimeRoundedTransformer()

        # assert
        assert dtrt.strip_constant
