from unittest import TestCase
from unittest.mock import Mock, patch

import copulas
import numpy as np
import pandas as pd
import pytest
from copulas import univariate

from rdt.transformers.null import NullTransformer
from rdt.transformers.numerical import (
    BayesGMMTransformer, GaussianCopulaTransformer, NumericalBoundedTransformer,
    NumericalRoundedBoundedTransformer, NumericalRoundedTransformer, NumericalTransformer)


class TestNumericalTransformer(TestCase):

    def test___init__super_attrs(self):
        """super() arguments are properly passed and set as attributes."""
        nt = NumericalTransformer(dtype='int', nan='mode', null_column=False)

        assert nt.dtype == 'int'
        assert nt.nan == 'mode'
        assert nt.null_column is False

    def test_get_output_types(self):
        """Test the ``get_output_types`` method when a null column is created.

        When a null column is created, this method should apply the ``_add_prefix``
        method to the following dictionary of output types:

        output_types = {
            'value': 'float',
            'is_null': 'float'
        }

        Setup:
            - initialize a ``NumericalTransformer`` transformer which:
                - sets ``self.null_transformer`` to a ``NullTransformer`` where
                ``self.null_column`` is True.
                - sets ``self.column_prefix`` to a string.

        Output:
            - the ``output_types`` dictionary, but with the ``self.column_prefix``
            added to the beginning of the keys.
        """
        # Setup
        transformer = NumericalTransformer()
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

    def test_is_composition_identity_null_transformer_true(self):
        """Test the ``is_composition_identity`` method with a ``null_transformer``.

        When the attribute ``null_transformer`` is not None and a null column is not created,
        this method should simply return False.

        Setup:
            - initialize a ``NumericalTransformer`` transformer which sets
            ``self.null_transformer`` to a ``NullTransformer`` where
            ``self.null_column`` is False.

        Output:
            - False
        """
        # Setup
        transformer = NumericalTransformer()
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
            - initialize a ``NumericalTransformer`` transformer which sets
            ``self.null_transformer`` to None.

        Output:
            - the value stored in ``self.COMPOSITION_IS_IDENTITY``.
        """
        # Setup
        transformer = NumericalTransformer()
        transformer.null_transformer = None

        # Run
        output = transformer.is_composition_identity()

        # Assert
        assert output is True

    def test__learn_rounding_digits_more_than_15_decimals(self):
        """Test the _learn_rounding_digits method with more than 15 decimals.

        If the data has more than 15 decimals, None should be returned.

        Input:
        - An array that contains floats with more than 15 decimals.
        Output:
        - None
        """
        data = np.random.random(size=10).round(20)

        output = NumericalTransformer._learn_rounding_digits(data)

        assert output is None

    def test__learn_rounding_digits_less_than_15_decimals(self):
        """Test the _learn_rounding_digits method with less than 15 decimals.

        If the data has less than 15 decimals, the maximum number of decimals
        should be returned.

        Input:
        - An array that contains floats with a maximum of 3 decimals and a NaN.
        Output:
        - 3
        """
        data = np.array([10, 0., 0.1, 0.12, 0.123, np.nan])

        output = NumericalTransformer._learn_rounding_digits(data)

        assert output == 3

    def test__learn_rounding_digits_negative_decimals_float(self):
        """Test the _learn_rounding_digits method with floats multiples of powers of 10.

        If the data has all multiples of 10, 100, or any other higher power of 10,
        the output is the negative number of decimals representing the corresponding
        power of 10.

        Input:
        - An array that contains floats that are multiples of powers of 10, 100 and 1000
          and a NaN.
        Output:
        - -1
        """
        data = np.array([1230., 12300., 123000., np.nan])

        output = NumericalTransformer._learn_rounding_digits(data)

        assert output == -1

    def test__learn_rounding_digits_negative_decimals_integer(self):
        """Test the _learn_rounding_digits method with integers multiples of powers of 10.

        If the data has all multiples of 10, 100, or any other higher power of 10,
        the output is the negative number of decimals representing the corresponding
        power of 10.

        Input:
        - An array that contains integers that are multiples of powers of 10, 100 and 1000
          and a NaN.
        Output:
        - -1
        """
        data = np.array([1230, 12300, 123000, np.nan])

        output = NumericalTransformer._learn_rounding_digits(data)

        assert output == -1

    def test__learn_rounding_digits_all_nans(self):
        """Test the _learn_rounding_digits method with data that is all NaNs.

        If the data is all NaNs, expect that the output is None.

        Input:
        - An array of NaNs.
        Output:
        - None
        """
        data = np.array([np.nan, np.nan, np.nan, np.nan])

        output = NumericalTransformer._learn_rounding_digits(data)

        assert output is None

    def test__fit(self):
        """Test the ``_fit`` method with numpy.array.

        Validate that the ``_dtype`` and ``.null_transformer.fill_value`` attributes
        are set correctly.

        Setup:
            - initialize a ``NumericalTransformer`` with the ``nan` parameter set to ``'nan'``.

        Input:
            - a pandas dataframe containing a None.

        Side effect:
            - it sets the ``null_transformer.fill_value``.
            - it sets the ``_dtype``.
        """
        # Setup
        data = pd.DataFrame([1.5, None, 2.5], columns=['a'])
        transformer = NumericalTransformer(dtype=float, nan='nan')

        # Run
        transformer._fit(data)

        # Asserts
        expect_fill_value = 'nan'
        assert transformer.null_transformer.fill_value == expect_fill_value
        expect_dtype = float
        assert transformer._dtype == expect_dtype

    def test__fit_rounding_none(self):
        """Test _fit rounding parameter with ``None``

        If the rounding parameter is set to ``None``, the ``_fit`` method
        should not set its ``rounding`` or ``_rounding_digits`` instance
        variables.

        Input:
        - An array with floats rounded to one decimal and a None value
        Side Effect:
        - ``rounding`` and ``_rounding_digits`` continue to be ``None``
        """
        # Setup
        data = pd.DataFrame([1.5, None, 2.5], columns=['a'])

        # Run
        transformer = NumericalTransformer(dtype=float, nan='nan', rounding=None)
        transformer._fit(data)

        # Asserts
        assert transformer.rounding is None
        assert transformer._rounding_digits is None

    def test__fit_rounding_int(self):
        """Test _fit rounding parameter with int

        If the rounding parameter is set to ``None``, the ``_fit`` method
        should not set its ``rounding`` or ``_rounding_digits`` instance
        variables.

        Input:
        - An array with floats rounded to one decimal and a None value
        Side Effect:
        - ``rounding`` and ``_rounding_digits`` are the provided int
        """
        # Setup
        data = pd.DataFrame([1.5, None, 2.5], columns=['a'])
        expected_digits = 3

        # Run
        transformer = NumericalTransformer(dtype=float, nan='nan', rounding=expected_digits)
        transformer._fit(data)

        # Asserts
        assert transformer.rounding == expected_digits
        assert transformer._rounding_digits == expected_digits

    def test__fit_rounding_auto(self):
        """Test _fit rounding parameter with ``'auto'``

        If the ``rounding`` parameter is set to ``'auto'``,
        ``_fit`` should learn the ``_rounding_digits`` to be the max
        number of decimal places seen in the data.

        Input:
        - Array of floats with up to 4 decimals
        Side Effect:
        - ``_rounding_digits`` is set to 4
        """
        # Setup
        data = pd.DataFrame([1, 2.1, 3.12, 4.123, 5.1234, 6.123, 7.12, 8.1, 9], columns=['a'])

        # Run
        transformer = NumericalTransformer(dtype=float, nan='nan', rounding='auto')
        transformer._fit(data)

        # Asserts
        assert transformer._rounding_digits == 4

    def test__fit_rounding_auto_large_numbers(self):
        """Test _fit rounding parameter with ``'auto'``

        If the ``rounding`` parameter is set to ``'auto'``
        and the data is very large, ``_fit`` should learn
        ``_rounding_digits`` to be the biggest number of 0s
        to round to that keeps the data the same.

        Input:
        - Array of data with numbers between 10^10 and 10^20
        Side Effect:
        - ``_rounding_digits`` is set to the minimum exponent seen in the data
        """
        # Setup
        exponents = [np.random.randint(10, 20) for i in range(10)]
        big_numbers = [10**exponents[i] for i in range(10)]
        data = pd.DataFrame(big_numbers, columns=['a'])

        # Run
        transformer = NumericalTransformer(dtype=float, nan='nan', rounding='auto')
        transformer._fit(data)

        # Asserts
        assert transformer._rounding_digits == -min(exponents)

    def test__fit_rounding_auto_max_decimals(self):
        """Test _fit rounding parameter with ``'auto'``

        If the ``rounding`` parameter is set to ``'auto'``,
        ``_fit`` should learn the ``_rounding_digits`` to be the max
        number of decimal places seen in the data. The max
        amount of decimals that floats can be accurately compared
        with is 15. If the input data has values with more than
        14 decimals, we will not be able to accurately learn the
        number of decimal places required, so we do not round.

        Input:
        - Array with a value that has 15 decimals
        Side Effect:
        - ``_rounding_digits`` is set to ``None``
        """
        # Setup
        data = pd.DataFrame([0.000000000000001], columns=['a'])

        # Run
        transformer = NumericalTransformer(dtype=float, nan='nan', rounding='auto')
        transformer._fit(data)

        # Asserts
        assert transformer._rounding_digits is None

    def test__fit_rounding_auto_max_inf(self):
        """Test _fit rounding parameter with ``'auto'``

        If the ``rounding`` parameter is set to ``'auto'``,
        and the data contains infinite values, ``_fit`` should
        learn the ``_rounding_digits`` to be the min
        number of decimal places seen in the data with
        the infinite values filtered out.

        Input:
        - Array with ``np.inf`` as a value
        Side Effect:
        - ``_rounding_digits`` is set to max seen in rest of data
        """
        # Setup
        data = pd.DataFrame([15000, 4000, 60000, np.inf], columns=['a'])

        # Run
        transformer = NumericalTransformer(dtype=float, nan='nan', rounding='auto')
        transformer._fit(data)

        # Asserts
        assert transformer._rounding_digits == -3

    def test__fit_rounding_auto_max_zero(self):
        """Test _fit rounding parameter with ``'auto'``

        If the ``rounding`` parameter is set to ``'auto'``,
        and the max in the data is 0, ``_fit`` should
        learn the ``_rounding_digits`` to be 0.

        Input:
        - Array with 0 as max value
        Side Effect:
        - ``_rounding_digits`` is set to 0
        """
        # Setup
        data = pd.DataFrame([0, 0, 0], columns=['a'])

        # Run
        transformer = NumericalTransformer(dtype=float, nan='nan', rounding='auto')
        transformer._fit(data)

        # Asserts
        assert transformer._rounding_digits == 0

    def test__fit_rounding_auto_max_negative(self):
        """Test _fit rounding parameter with ``'auto'``

        If the ``rounding`` parameter is set to ``'auto'``,
        and the max in the data is negative, the ``_fit`` method
        should learn ``_rounding_digits`` to be the min number
        of digits seen in those negative values.

        Input:
        - Array with negative max value
        Side Effect:
        - ``_rounding_digits`` is set to min number of digits in array
        """
        # Setup
        data = pd.DataFrame([-500, -220, -10], columns=['a'])

        # Run
        transformer = NumericalTransformer(dtype=float, nan='nan', rounding='auto')
        transformer._fit(data)

        # Asserts
        assert transformer._rounding_digits == -1

    def test__fit_min_max_none(self):
        """Test _fit min and max parameters with ``None``

        If the min and max parameters are set to ``None``,
        the ``_fit`` method should not set its ``min`` or ``max``
        instance variables.

        Input:
        - Array of floats and null values
        Side Effect:
        - ``_min_value`` and ``_max_value`` stay ``None``
        """
        # Setup
        data = pd.DataFrame([1.5, None, 2.5], columns=['a'])

        # Run
        transformer = NumericalTransformer(dtype=float, nan='nan',
                                           min_value=None, max_value=None)
        transformer._fit(data)

        # Asserts
        assert transformer._min_value is None
        assert transformer._max_value is None

    def test__fit_min_max_int(self):
        """Test _fit min and max parameters with int values

        If the min and max parameters are set to an int,
        the ``_fit`` method should not change them.

        Input:
        - Array of floats and null values
        Side Effect:
        - ``_min_value`` and ``_max_value`` remain unchanged
        """
        # Setup
        data = pd.DataFrame([1.5, None, 2.5], columns=['a'])

        # Run
        transformer = NumericalTransformer(dtype=float, nan='nan',
                                           min_value=1, max_value=10)
        transformer._fit(data)

        # Asserts
        assert transformer._min_value == 1
        assert transformer._max_value == 10

    def test__fit_min_max_auto(self):
        """Test _fit min and max parameters with ``'auto'``

        If the min or max parameters are set to ``'auto'``
        the ``_fit`` method should learn them from the
        _fitted data.

        Input:
        - Array of floats and null values
        Side Effect:
        - ``_min_value`` and ``_max_value`` are learned
        """
        # Setup
        data = pd.DataFrame([-100, -5000, 0, None, 100, 4000], columns=['a'])

        # Run
        transformer = NumericalTransformer(dtype=float, nan='nan',
                                           min_value='auto', max_value='auto')
        transformer._fit(data)

        # Asserts
        assert transformer._min_value['a'] == -5000
        assert transformer._max_value['a'] == 4000

    def test__transform(self):
        """Test the ``_transform`` method.

        Validate that this method calls the ``self.null_transformer.transform`` method once.

        Setup:
            - create an instance of a ``NumericalTransformer`` and set ``self.null_transformer``
            to a ``NullTransformer``.

        Input:
            - a pandas series.

        Output:
            - the transformed numpy array.
        """
        # Setup
        data = pd.Series([1, 2, 3])
        transformer = NumericalTransformer()
        transformer.null_transformer = Mock()

        # Run
        transformer._transform(data)

        # Assert
        assert transformer.null_transformer.transform.call_count == 1

    def test__reverse_transform_rounding_none(self):
        """Test ``_reverse_transform`` when ``rounding`` is ``None``

        The data should not be rounded at all.

        Input:
        - Random array of floats between 0 and 1
        Output:
        - Input array
        """
        # Setup
        data = np.random.random(10)

        # Run
        transformer = NumericalTransformer(dtype=float, nan=None)
        transformer._rounding_digits = None
        result = transformer._reverse_transform(data)

        # Assert
        np.testing.assert_array_equal(result, data)

    def test__reverse_transform_rounding_none_integer(self):
        """Test ``_reverse_transform`` when ``rounding`` is ``None`` and the dtype is integer.

        The data should be rounded to 0 decimals and returned as integer values.

        Input:
        - Array of multiple float values with decimals.
        Output:
        - Input array rounded an converted to integers.
        """
        # Setup
        data = np.array([0., 1.2, 3.45, 6.789])

        # Run
        transformer = NumericalTransformer(dtype=np.int64, nan=None)
        transformer._rounding_digits = None
        transformer._dtype = np.int64
        result = transformer._reverse_transform(data)

        # Assert
        expected = np.array([0, 1, 3, 7])
        np.testing.assert_array_equal(result, expected)

    def test__reverse_transform_rounding_none_with_nulls(self):
        """Test ``_reverse_transform`` when ``rounding`` is ``None`` and there are nulls.

        The data should not be rounded at all.

        Input:
        - 2d Array of multiple float values with decimals and a column setting at least 1 null.
        Output:
        - First column of the input array as entered, replacing the indicated value with a Nan.
        """
        # Setup
        data = [
            [0., 0.],
            [1.2, 0.],
            [3.45, 1.],
            [6.789, 0.],
        ]

        data = pd.DataFrame(data, columns=['a', 'b'])

        # Run
        transformer = NumericalTransformer()
        null_transformer = Mock()
        null_transformer.reverse_transform.return_value = np.array([0., 1.2, np.nan, 6.789])
        transformer.null_transformer = null_transformer
        transformer._rounding_digits = None
        transformer._dtype = float
        result = transformer._reverse_transform(data)

        # Assert
        expected = np.array([0., 1.2, np.nan, 6.789])
        np.testing.assert_array_equal(result, expected)

    def test__reverse_transform_rounding_none_with_nulls_dtype_int(self):
        """Test ``_reverse_transform`` when rounding is None, dtype is int and there are nulls.

        The data should be rounded to 0 decimals and returned as float values with
        nulls in the right place.

        Input:
        - 2d Array of multiple float values with decimals and a column setting at least 1 null.
        Output:
        - First column of the input array rounded, replacing the indicated value with a Nan,
          and kept as float values.
        """
        # Setup
        data = np.array([
            [0., 0.],
            [1.2, 0.],
            [3.45, 1.],
            [6.789, 0.],
        ])

        # Run
        transformer = NumericalTransformer()
        null_transformer = Mock()
        null_transformer.reverse_transform.return_value = np.array([0., 1.2, np.nan, 6.789])
        transformer.null_transformer = null_transformer
        transformer._rounding_digits = None
        transformer._dtype = int
        result = transformer._reverse_transform(data)

        # Assert
        expected = np.array([0., 1., np.nan, 7.])
        np.testing.assert_array_equal(result, expected)

    def test__reverse_transform_rounding_positive_rounding(self):
        """Test ``_reverse_transform`` when ``rounding`` is a positive int

        The data should round to the maximum number of decimal places
        set in the ``_rounding_digits`` value.

        Input:
        - Array with decimals

        Output:
        - Same array rounded to the provided number of decimal places
        """
        # Setup
        data = np.array([1.1111, 2.2222, 3.3333, 4.44444, 5.555555])

        # Run
        transformer = NumericalTransformer(dtype=float, nan=None)
        transformer._rounding_digits = 2
        result = transformer._reverse_transform(data)

        # Assert
        expected_data = np.array([1.11, 2.22, 3.33, 4.44, 5.56])
        np.testing.assert_array_equal(result, expected_data)

    def test__reverse_transform_rounding_negative_rounding_int(self):
        """Test ``_reverse_transform`` when ``rounding`` is a negative int

        The data should round to the number set in the ``_rounding_digits``
        attribute and remain ints.

        Input:
        - Array with with floats above 100

        Output:
        - Same array rounded to the provided number of 0s
        - Array should be of type int
        """
        # Setup
        data = np.array([2000.0, 120.0, 3100.0, 40100.0])

        # Run
        transformer = NumericalTransformer(dtype=int, nan=None)
        transformer._dtype = int
        transformer._rounding_digits = -3
        result = transformer._reverse_transform(data)

        # Assert
        expected_data = np.array([2000, 0, 3000, 40000])
        np.testing.assert_array_equal(result, expected_data)
        assert result.dtype == int

    def test__reverse_transform_rounding_negative_rounding_float(self):
        """Test ``_reverse_transform`` when ``rounding`` is a negative int

        The data should round to the number set in the ``_rounding_digits``
        attribute and remain floats.

        Input:
        - Array with with larger numbers

        Output:
        - Same array rounded to the provided number of 0s
        - Array should be of type float
        """
        # Setup
        data = np.array([2000.0, 120.0, 3100.0, 40100.0])

        # Run
        transformer = NumericalTransformer(dtype=float, nan=None)
        transformer._rounding_digits = -3
        result = transformer._reverse_transform(data)

        # Assert
        expected_data = np.array([2000.0, 0.0, 3000.0, 40000.0])
        np.testing.assert_array_equal(result, expected_data)
        assert result.dtype == float

    def test__reverse_transform_rounding_zero(self):
        """Test ``_reverse_transform`` when ``rounding`` is a negative int

        The data should round to the number set in the ``_rounding_digits``
        attribute.

        Input:
        - Array with with larger numbers

        Output:
        - Same array rounded to the provided number of 0s
        """
        # Setup
        data = np.array([2000.554, 120.2, 3101, 4010])

        # Run
        transformer = NumericalTransformer(dtype=float, nan=None)
        transformer._rounding_digits = 0
        result = transformer._reverse_transform(data)

        # Assert
        expected_data = np.array([2001, 120, 3101, 4010])
        np.testing.assert_array_equal(result, expected_data)

    def test__reverse_transform_min_no_max(self):
        """Test _reverse_transform with ``min_value`` set

        The ``_reverse_transform`` method should clip any values below
        the ``min_value`` if it is set.

        Input:
        - Array with values below the min and infinitely high values
        Output:
        - Array with low values clipped to min
        """
        # Setup
        data = np.array([-np.inf, -5000, -301, -250, 0, 125, 400, np.inf])

        # Run
        transformer = NumericalTransformer(dtype=float, nan=None)
        transformer._min_value = -300
        result = transformer._reverse_transform(data)

        # Asserts
        expected_data = np.array([-300, -300, -300, -250, 0, 125, 400, np.inf])
        np.testing.assert_array_equal(result, expected_data)

    def test__reverse_transform_max_no_min(self):
        """Test _reverse_transform with ``max_value`` set

        The ``_reverse_transform`` method should clip any values above
        the ``max_value`` if it is set.

        Input:
        - Array with values above the max and infinitely low values
        Output:
        - Array with values clipped to max
        """
        # Setup
        data = np.array([-np.inf, -5000, -301, -250, 0, 125, 401, np.inf])

        # Run
        transformer = NumericalTransformer(dtype=float, nan=None)
        transformer._max_value = 400
        result = transformer._reverse_transform(data)

        # Asserts
        expected_data = np.array([-np.inf, -5000, -301, -250, 0, 125, 400, 400])
        np.testing.assert_array_equal(result, expected_data)

    def test__reverse_transform_min_and_max(self):
        """Test _reverse_transform with ``min_value`` and ``max_value`` set

        The ``_reverse_transform`` method should clip any values above
        the ``max_value`` and any values below the ``min_value``.

        Input:
        - Array with values above the max and below the min
        Output:
        - Array with out of bound values clipped to min and max
        """
        # Setup
        data = np.array([-np.inf, -5000, -301, -250, 0, 125, 401, np.inf])

        # Run
        transformer = NumericalTransformer(dtype=float, nan=None)
        transformer._max_value = 400
        transformer._min_value = -300
        result = transformer._reverse_transform(data)

        # Asserts
        np.testing.assert_array_equal(result, np.array([-300, -300, -300, -250, 0, 125, 400, 400]))

    def test__reverse_transform_min_an_max_with_nulls(self):
        """Test _reverse_transform with nulls and ``min_value`` and ``max_value`` set

        The ``_reverse_transform`` method should clip any values above
        the ``max_value`` and any values below the ``min_value``. Null values
        should be replaced with ``np.nan``.

        Input:
        - 2d array where second column has some values over 0.5 representing null values
        Output:
        - Array with out of bounds values clipped and null values injected
        """
        # Setup
        data = np.array([
            [-np.inf, 0],
            [-5000, 0.1],
            [-301, 0.8],
            [-250, 0.4],
            [0, 0],
            [125, 1],
            [401, 0.2],
            [np.inf, 0.5]
        ])
        clipped_data = np.array([
            [-300, 0],
            [-300, 0.1],
            [-300, 0.8],
            [-250, 0.4],
            [0, 0],
            [125, 1],
            [400, 0.2],
            [400, 0.5]
        ])
        expected_data = np.array([-300, -300, np.nan, -250, 0, np.nan, 400, 400])

        # Run
        transformer = NumericalTransformer(dtype=float, nan='nan')
        transformer._max_value = 400
        transformer._min_value = -300
        transformer.null_transformer = Mock()
        transformer.null_transformer.reverse_transform.return_value = expected_data
        result = transformer._reverse_transform(data)

        # Asserts
        null_transformer_calls = transformer.null_transformer.reverse_transform.mock_calls
        np.testing.assert_array_equal(null_transformer_calls[0][1][0], clipped_data)
        np.testing.assert_array_equal(result, expected_data)


class TestNumericalBoundedTransformer(TestCase):

    def test___init__(self):
        """super() arguments are properly passed and set as attributes."""
        # Run
        nt = NumericalBoundedTransformer(dtype='int', null_column=False)

        # Assert
        assert nt.dtype == 'int'
        assert nt.nan == 'mean'
        assert nt.null_column is False
        assert nt.min_value == 'auto'
        assert nt.max_value == 'auto'
        assert nt.rounding is None


class TestNumericalRoundedTransformer(TestCase):

    def test___init__(self):
        """super() arguments are properly passed and set as attributes."""
        # Run
        nt = NumericalRoundedTransformer(dtype='int', null_column=False)

        # Assert
        assert nt.dtype == 'int'
        assert nt.nan == 'mean'
        assert nt.null_column is False
        assert nt.min_value is None
        assert nt.max_value is None
        assert nt.rounding == 'auto'


class TestNumericalRoundedBoundedTransformer(TestCase):

    def test___init__(self):
        """super() arguments are properly passed and set as attributes."""
        # Run
        nt = NumericalRoundedBoundedTransformer(dtype='int', null_column=False)

        # Assert
        assert nt.dtype == 'int'
        assert nt.nan == 'mean'
        assert nt.null_column is False
        assert nt.min_value == 'auto'
        assert nt.max_value == 'auto'
        assert nt.rounding == 'auto'


class TestGaussianCopulaTransformer:

    def test___init__super_attrs(self):
        """super() arguments are properly passed and set as attributes."""
        ct = GaussianCopulaTransformer(dtype='int', nan='mode', null_column=False)

        assert ct.dtype == 'int'
        assert ct.nan == 'mode'
        assert ct.null_column is False

    def test___init__str_distr(self):
        """If distribution is an str, it is resolved using the _DISTRIBUTIONS dict."""
        ct = GaussianCopulaTransformer(distribution='univariate')

        assert ct._distribution is copulas.univariate.Univariate

    def test___init__non_distr(self):
        """If distribution is not an str, it is store as given."""
        univariate = copulas.univariate.Univariate()
        ct = GaussianCopulaTransformer(distribution=univariate)

        assert ct._distribution is univariate

    def test__get_distributions_copulas_not_installed(self):
        """Test the ``_get_distributions`` method when copulas is not installed.

        Validate that this method raises the appropriate error message when copulas is
        not installed.

        Raise:
            - ImportError('\n\nIt seems like `copulas` is not installed.\n'
            'Please install it using:\n\n    pip install rdt[copulas]')
        """
        __py_import__ = __import__

        def custom_import(name, *args):
            if name == 'copulas':
                raise ImportError('Simulate copulas not being importable.')

            return __py_import__(name, *args)

        with patch('builtins.__import__', side_effect=custom_import):
            with pytest.raises(ImportError, match=r'pip install rdt\[copulas\]'):
                GaussianCopulaTransformer._get_distributions()

    def test__get_distributions(self):
        """Test the ``_get_distributions`` method.

        Validate that this method returns the correct dictionary of distributions.

        Setup:
            - instantiate a ``GaussianCopulaTransformer``.
        """
        # Setup
        transformer = GaussianCopulaTransformer()

        # Run
        distributions = transformer._get_distributions()

        # Assert
        expected = {
            'univariate': univariate.Univariate,
            'parametric': (
                univariate.Univariate, {
                    'parametric': univariate.ParametricType.PARAMETRIC,
                },
            ),
            'bounded': (
                univariate.Univariate,
                {
                    'bounded': univariate.BoundedType.BOUNDED,
                },
            ),
            'semi_bounded': (
                univariate.Univariate,
                {
                    'bounded': univariate.BoundedType.SEMI_BOUNDED,
                },
            ),
            'parametric_bounded': (
                univariate.Univariate,
                {
                    'parametric': univariate.ParametricType.PARAMETRIC,
                    'bounded': univariate.BoundedType.BOUNDED,
                },
            ),
            'parametric_semi_bounded': (
                univariate.Univariate,
                {
                    'parametric': univariate.ParametricType.PARAMETRIC,
                    'bounded': univariate.BoundedType.SEMI_BOUNDED,
                },
            ),
            'gaussian': univariate.GaussianUnivariate,
            'gamma': univariate.GammaUnivariate,
            'beta': univariate.BetaUnivariate,
            'student_t': univariate.StudentTUnivariate,
            'gaussian_kde': univariate.GaussianKDE,
            'truncated_gaussian': univariate.TruncatedGaussian,
        }
        assert distributions == expected

    def test__get_univariate_instance(self):
        """Test the ``_get_univariate`` method when the distribution is univariate.

        Validate that a deepcopy of the distribution stored in ``self._distribution`` is returned.

        Setup:
            - create an instance of a ``GaussianCopulaTransformer`` with ``distribution`` set
            to ``univariate.Univariate``.

        Output:
            - a copy of the value stored in ``self._distribution``.
        """
        # Setup
        distribution = copulas.univariate.Univariate()
        ct = GaussianCopulaTransformer(distribution=distribution)

        # Run
        univariate = ct._get_univariate()

        # Assert
        assert univariate is not distribution
        assert isinstance(univariate, copulas.univariate.Univariate)
        assert dir(univariate) == dir(distribution)

    def test__get_univariate_tuple(self):
        """Test the ``_get_univariate`` method when the distribution is a tuple.

        When the distribution is passed as a tuple, it should return an instance
        with the passed arguments.

        Setup:
            - create an instance of a ``GaussianCopulaTransformer`` and set
            ``distribution`` to a tuple.

        Output:
            - an instance of ``copulas.univariate.Univariate`` with the passed arguments.
        """
        # Setup
        distribution = (
            copulas.univariate.Univariate,
            {'candidates': 'a_candidates_list'}
        )
        ct = GaussianCopulaTransformer(distribution=distribution)

        # Run
        univariate = ct._get_univariate()

        # Assert
        assert isinstance(univariate, copulas.univariate.Univariate)
        assert univariate.candidates == 'a_candidates_list'

    def test__get_univariate_class(self):
        """Test the ``_get_univariate`` method when the distribution is a class.

        When ``distribution`` is passed as a class, it should return an instance
        without passing arguments.

        Setup:
            - create an instance of a ``GaussianCopulaTransformer`` and set ``distribution``
            to ``univariate.Univariate``.

        Output:
            - an instance of ``copulas.univariate.Univariate`` without any arguments.
        """
        # Setup
        distribution = copulas.univariate.Univariate
        ct = GaussianCopulaTransformer(distribution=distribution)

        # Run
        univariate = ct._get_univariate()

        # Assert
        assert isinstance(univariate, copulas.univariate.Univariate)

    def test__get_univariate_error(self):
        """Test the ``_get_univariate`` method when ``distribution`` is invalid.

        Validate that it raises an error if an invalid distribution is stored in
        ``distribution``.

        Setup:
            - create an instance of a ``GaussianCopulaTransformer`` and set ``self._distribution``
            improperly.

        Raise:
            - TypeError(f'Invalid distribution: {distribution}')
        """
        # Setup
        distribution = 123
        ct = GaussianCopulaTransformer(distribution=distribution)

        # Run / Assert
        with pytest.raises(TypeError):
            ct._get_univariate()

    def test__fit(self):
        """Test the ``_fit`` method.

        Validate that ``_fit`` calls ``_get_univariate``.

        Setup:
            - create an instance of the ``GaussianCopulaTransformer``.
            - mock the  ``_get_univariate`` method.

        Input:
            - a pandas series of float values.

        Side effect:
            - call the `_get_univariate`` method.
        """
        # Setup
        data = pd.Series([0.0, np.nan, 1.0])
        ct = GaussianCopulaTransformer()
        ct._get_univariate = Mock()

        # Run
        ct._fit(data)

        # Assert
        ct._get_univariate.return_value.fit.assert_called_once()
        call_value = ct._get_univariate.return_value.fit.call_args_list[0]
        np.testing.assert_array_equal(
            call_value[0][0],
            np.array([0.0, 0.5, 1.0])
        )

    def test__copula_transform(self):
        """Test the ``_copula_transform`` method.

        Validate that ``_copula_transform`` calls ``_get_univariate``.

        Setup:
            - create an instance of the ``GaussianCopulaTransformer``.
            - mock  ``_univariate``.

        Input:
            - a pandas series of float values.

        Ouput:
            - a numpy array of the transformed data.
        """
        # Setup
        ct = GaussianCopulaTransformer()
        ct._univariate = Mock()
        ct._univariate.cdf.return_value = np.array([0.25, 0.5, 0.75])
        data = pd.Series([0.0, 1.0, 2.0])

        # Run
        transformed_data = ct._copula_transform(data)

        # Assert
        expected = np.array([-0.67449, 0, 0.67449])
        np.testing.assert_allclose(transformed_data, expected, rtol=1e-3)

    def test__transform(self):
        """Test the ``_transform`` method.

        Validate that ``_transform`` produces the correct values when ``null_column`` is True.

        Setup:
            - create an instance of the ``GaussianCopulaTransformer``, where:
                - ``self._univariate`` is a mock.
                - ``self.null_transformer``  is a ``NullTransformer``.
                - fit the ``self.null_transformer``.

        Input:
            - a pandas series of float values.

        Ouput:
            - a numpy array of the transformed data.
        """
        # Setup
        data = pd.Series([0.0, 1.0, 2.0, np.nan])
        ct = GaussianCopulaTransformer()
        ct._univariate = Mock()
        ct._univariate.cdf.return_value = np.array([0.25, 0.5, 0.75, 0.5])
        ct.null_transformer = NullTransformer(None, null_column=True)
        ct.null_transformer.fit(data)

        # Run
        transformed_data = ct._transform(data)

        # Assert
        expected = np.array([
            [-0.67449, 0, 0.67449, 0],
            [0, 0, 0, 1.0]
        ]).T
        np.testing.assert_allclose(transformed_data, expected, rtol=1e-3)

    def test__transform_null_column_none(self):
        """Test the ``_transform`` method.

        Validate that ``_transform`` produces the correct values when ``null_column`` is None.

        Setup:
            - create an instance of the ``GaussianCopulaTransformer``, where:
                - ``self._univariate`` is a mock.
                - ``self.null_transformer``  is a ``NullTransformer``.
                - fit the ``self.null_transformer``.

        Input:
            - a pandas series of float values.

        Ouput:
            - a numpy array of the transformed data.
        """
        # Setup
        data = pd.Series([
            0.0, 1.0, 2.0, 1.0
        ])
        ct = GaussianCopulaTransformer()
        ct._univariate = Mock()
        ct._univariate.cdf.return_value = np.array([0.25, 0.5, 0.75, 0.5])
        ct.null_transformer = NullTransformer(None, null_column=None)

        # Run
        ct.null_transformer.fit(data)
        transformed_data = ct._transform(data)

        # Assert
        expected = np.array([-0.67449, 0, 0.67449, 0]).T
        np.testing.assert_allclose(transformed_data, expected, rtol=1e-3)

    def test__reverse_transform(self):
        """Test the ``_reverse_transform`` method.

        Validate that ``_reverse_transform`` produces the correct values when
        ``null_column`` is True.

        Setup:
            - create an instance of the ``GaussianCopulaTransformer``, where:
                - ``self._univariate`` is a mock.
                - ``self.null_transformer``  is a ``NullTransformer``.
                - fit the ``self.null_transformer``.

        Input:
            - a pandas series of float values.

        Ouput:
            - a numpy array of the transformed data.
        """
        # Setup
        data = np.array([
            [-0.67449, 0, 0.67449, 0],
            [0, 0, 0, 1.0],
        ]).T
        expected = pd.Series([
            0.0, 1.0, 2.0, np.nan
        ])
        ct = GaussianCopulaTransformer()
        ct._univariate = Mock()
        ct._univariate.ppf.return_value = np.array([0.0, 1.0, 2.0, 1.0])
        ct.null_transformer = NullTransformer(None, null_column=True)

        # Run
        ct.null_transformer.fit(expected)
        transformed_data = ct._reverse_transform(data)

        # Assert
        np.testing.assert_allclose(transformed_data, expected, rtol=1e-3)

    def test__reverse_transform_null_column_none(self):
        """Test the ``_reverse_transform`` method.

        Validate that ``_reverse_transform`` produces the correct values when
        ``null_column`` is None.

        Setup:
            - create an instance of the ``GaussianCopulaTransformer``, where:
                - ``self._univariate`` is a mock.
                - ``self.null_transformer``  is a ``NullTransformer``.
                - fit the ``self.null_transformer``.

        Input:
            - a pandas series of float values.

        Ouput:
            - a numpy array of the transformed data.
        """
        # Setup
        data = pd.Series(
            [-0.67449, 0, 0.67449, 0]
        ).T
        expected = pd.Series([
            0.0, 1.0, 2.0, 1.0
        ])
        ct = GaussianCopulaTransformer()
        ct._univariate = Mock()
        ct._univariate.ppf.return_value = np.array([0.0, 1.0, 2.0, 1.0])
        ct.null_transformer = NullTransformer(None, null_column=None)

        # Run
        ct.null_transformer.fit(expected)
        transformed_data = ct._reverse_transform(data)

        # Assert
        np.testing.assert_allclose(transformed_data, expected, rtol=1e-3)


class TestBayesGMMTransformer(TestCase):

    def test_get_output_types_null_column_created(self):
        """Test the ``get_output_types`` method when a null column is created.

        When a null column is created, this method should apply the ``_add_prefix``
        method to the following dictionary of output types:

        output_types = {
            'value': 'float',
            'is_null': 'float'
        }

        Setup:
            - initialize a ``NumericalTransformer`` transformer which:
                - sets ``self.null_transformer`` to a ``NullTransformer`` where
                ``self._null_column`` is True.
                - sets ``self.column_prefix`` to a string.

        Output:
            - the ``output_types`` dictionary, but with ``self.column_prefix``
            added to the beginning of the keys.
        """
        # Setup
        transformer = BayesGMMTransformer()
        transformer.null_transformer = NullTransformer(fill_value='fill')
        transformer.null_transformer._null_column = True
        transformer.column_prefix = 'abc'

        # Run
        output = transformer.get_output_types()

        # Assert
        expected = {
            'abc.normalized': 'float',
            'abc.component': 'categorical',
            'abc.is_null': 'float'
        }
        assert output == expected

    @patch('rdt.transformers.numerical.BayesianGaussianMixture')
    def test__fit(self, mock_bgm):
        """Test ``_fit``.

        Validate that the method sets the internal variables to the correct values
        when given a pandas Series.

        Setup:
            - patch a ``BayesianGaussianMixture`` with ``weights_`` containing two components
            greater than the ``weight_threshold`` parameter.
            - create an instance of the ``BayesGMMTransformer``.

        Input:
            - a pandas Series containing random values.

        Side Effects:
            - the sum of ``_valid_component_indicator`` should equal to 2
            (the number of ``weights_`` greater than the threshold).
        """
        # Setup
        bgm_instance = mock_bgm.return_value
        bgm_instance.weights_ = np.array([10.0, 5.0, 0.0])
        transformer = BayesGMMTransformer(max_clusters=10, weight_threshold=0.005)
        data = pd.Series(np.random.random(size=100))

        # Run
        transformer._fit(data)

        # Asserts
        assert transformer._bgm_transformer == bgm_instance
        assert transformer._valid_component_indicator.sum() == 2

    @patch('rdt.transformers.numerical.BayesianGaussianMixture')
    def test__fit_nan(self, mock_bgm):
        """Test ``_fit`` with ``np.nan`` values.

        Validate that the method sets the internal variables to the correct values
        when given a pandas Series containing ``np.nan`` values.

        Setup:
            - patch a ``BayesianGaussianMixture`` with ``weights_`` containing two components
            greater than the ``weight_threshold`` parameter.
            - create an instance of the ``BayesGMMTransformer``.

        Input:
            - a pandas Series containing some ``np.nan`` values.

        Side Effects:
            - the sum of ``_valid_component_indicator`` should equal to 2
            (the number of ``weights_`` greater than the threshold).
            - set the ``null_transformer`` appropriately.
        """
        # Setup
        bgm_instance = mock_bgm.return_value
        bgm_instance.weights_ = np.array([10.0, 5.0, 0.0])
        transformer = BayesGMMTransformer(max_clusters=10, weight_threshold=0.005)

        data = pd.Series(np.random.random(size=100))
        mask = np.random.choice([1, 0], data.shape, p=[.1, .9]).astype(bool)
        data[mask] = np.nan

        # Run
        transformer._fit(data)

        # Asserts
        assert transformer._bgm_transformer == bgm_instance
        assert transformer._valid_component_indicator.sum() == 2
        assert transformer.null_transformer.creates_null_column()

    def test__transform(self):
        """Test ``_transform``.

        Validate that the method produces the appropriate output when given a pandas Series.

        Setup:
            - create an instance of the ``BayesGMMTransformer`` where:
                - ``_bgm_transformer`` is mocked with the appropriate ``means_``, ``covariances_``
                and ``predict_proba.return_value``.
                - ``_valid_component_indicator`` is set to ``np.array([True, True, False])``.

        Input:
            - a pandas Series.

        Ouput:
            - a numpy array with the transformed data.
        """
        # Setup
        np.random.seed(10)
        transformer = BayesGMMTransformer(max_clusters=3, nan=None)
        transformer._bgm_transformer = Mock()

        means = np.array([
            [0.90138867],
            [0.09169366],
            [0.499]
        ])
        transformer._bgm_transformer.means_ = means

        covariances = np.array([
            [[0.09024532]],
            [[0.08587948]],
            [[0.27487667]]
        ])
        transformer._bgm_transformer.covariances_ = covariances

        probabilities = np.array([
            [0.01519528, 0.98480472, 0.],
            [0.01659093, 0.98340907, 0.],
            [0.012744, 0.987256, 0.],
            [0.012744, 0.987256, 0.],
            [0.01391614, 0.98608386, 0.],
            [0.99220664, 0.00779336, 0.],
            [0.99059634, 0.00940366, 0.],
            [0.9941256, 0.0058744, 0.],
            [0.99465502, 0.00534498, 0.],
            [0.99059634, 0.00940366, 0.]
        ])
        transformer._bgm_transformer.predict_proba.return_value = probabilities

        transformer._valid_component_indicator = np.array([True, True, False])
        transformer.null_transformer = NullTransformer()
        data = pd.Series([0.01, 0.02, -0.01, -0.01, 0.0, 0.99, 0.97, 1.02, 1.03, 0.97])

        # Run
        output = transformer._transform(data)

        # Asserts
        assert output.shape == (10, 2)

        expected_normalized = np.array([
            -0.06969212, -0.06116121, -0.08675394, -0.08675394, -0.07822303,
            0.07374234, 0.05709835, 0.09870834, 0.10703034, 0.05709835
        ])
        np.testing.assert_allclose(output[:, 0], expected_normalized, rtol=1e-3)

        expected_component = np.array([1., 1., 1., 1., 1., 0., 0., 0., 0., 0.])
        np.testing.assert_allclose(output[:, 1], expected_component)

    def test__transform_nan(self):
        """Test ``_transform`` with ``np.nan`` values.

        Validate that the method produces the appropriate output when given a pandas Series
        containing ``np.nan`` values.

        Setup:
            - create an instance of the ``BayesGMMTransformer`` where:
                - ``_bgm_transformer`` is mocked with the appropriate ``means_``, ``covariances_``
                and ``predict_proba.return_value``.
                - ``_valid_component_indicator`` is set to ``np.array([True, True, False])``.
                - ``null_transformer`` is set to ``NullTransformer(0.0, True)``.

        Input:
            - a pandas Series.

        Ouput:
            - a numpy array with the transformed data.
        """
        # Setup
        np.random.seed(10)
        transformer = BayesGMMTransformer(nan=0.0, max_clusters=3)
        transformer._bgm_transformer = Mock()

        means = np.array([
            [0.03610001],
            [0.77135278],
            [0.292]
        ])
        transformer._bgm_transformer.means_ = means

        covariances = np.array([
            [[0.03819894]],
            [[0.16408241]],
            [[0.22328444]]
        ])
        transformer._bgm_transformer.covariances_ = covariances

        probabilities = np.array([
            [9.73559141e-01, 2.64408588e-02, 0.0],
            [9.74533917e-01, 2.54660826e-02, 0.0],
            [9.75425565e-01, 2.45744346e-02, 0.0],
            [9.75425565e-01, 2.45744346e-02, 0.0],
            [9.74533917e-01, 2.54660826e-02, 0.0],
            [4.93725426e-05, 9.99950627e-01, 0.0],
            [7.88963658e-05, 9.99921104e-01, 0.0],
            [9.74533917e-01, 2.54660826e-02, 0.0],
            [9.74533917e-01, 2.54660826e-02, 0.0],
            [7.88963658e-05, 9.99921104e-01, 0.0]
        ])
        transformer._bgm_transformer.predict_proba.return_value = probabilities

        transformer._valid_component_indicator = np.array([True, True, False])
        transformer.null_transformer = NullTransformer(0.0, null_column=True)
        data = pd.Series([0.01, np.nan, -0.01, -0.01, 0.0, 0.99, 0.97, np.nan, np.nan, 0.97])

        # Run
        transformer.null_transformer.fit(data)
        output = transformer._transform(data)

        # Asserts
        assert output.shape == (10, 3)

        expected_normalized = np.array([
            -0.033385, -0.046177, -0.058968, -0.058968, -0.046177,
            0.134944, 0.1226, -0.046177, -0.046177, 0.1226
        ])
        np.testing.assert_allclose(output[:, 0], expected_normalized, rtol=1e-3)

        expected_component = np.array([0., 0., 0., 0., 0., 1., 1., 0., 0., 1.])
        np.testing.assert_allclose(output[:, 1], expected_component)

        expected_null = np.array([0., 1., 0., 0., 0., 0., 0., 1., 1., 0.])
        np.testing.assert_allclose(output[:, 2], expected_null)

    def test__reverse_transform_helper(self):
        """Test ``_reverse_transform_helper``.

        Validate that the method produces the appropriate output when passed a numpy array.

        Setup:
            - create an instance of the ``BayesGMMTransformer`` where:
                - ``_bgm_transformer`` is mocked with the appropriate
                ``means_`` and ``covariances_``.
                - ``_valid_component_indicator`` is set to ``np.array([True, True, False])``.

        Input:
            - a numpy array containing the data to be reversed.

        Ouput:
            - a numpy array with the transformed data.
        """
        # Setup
        transformer = BayesGMMTransformer(max_clusters=3)
        transformer._bgm_transformer = Mock()

        means = np.array([
            [0.90138867],
            [0.09169366],
            [0.499]
        ])
        transformer._bgm_transformer.means_ = means

        covariances = np.array([
            [[0.09024532]],
            [[0.08587948]],
            [[0.27487667]]
        ])
        transformer._bgm_transformer.covariances_ = covariances

        transformer._valid_component_indicator = np.array([True, True, False])
        data = np.array([
            [-0.069, -0.061, -0.086, -0.086, -0.078, 0.073, 0.057, 0.098, 0.107, 0.057],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        ]).transpose()

        # Run
        output = transformer._reverse_transform_helper(data)

        # Asserts
        expected = pd.Series([0.01, 0.02, -0.01, -0.01, 0.0, 0.99, 0.97, 1.02, 1.03, 0.97])
        np.testing.assert_allclose(output, expected, atol=1e-3)

    def test__reverse_transform(self):
        """Test ``_reverse_transform``.

        Validate that the method correctly calls ``_reverse_transform_helper`` and produces the
        appropriate output when passed pandas dataframe.

        Setup:
            - create an instance of the ``BayesGMMTransformer`` where the ``output_columns``
            is a list of two columns.
            - mock the `_reverse_transform_helper` with the appropriate return value.

        Input:
            - a dataframe containing the data to be reversed.

        Ouput:
            - a pandas Series with the reverse transformed data.

        Side Effects:
            - ``_reverse_transform_helper`` should be called once with the correct data.
        """
        # Setup
        transformer = BayesGMMTransformer(max_clusters=3, nan=None)
        transformer.output_columns = ['col.normalized', 'col.component']
        transformer._reverse_transform_helper = Mock()
        transformer._reverse_transform_helper.return_value = np.array(
            [0.01, 0.02, -0.01, -0.01, 0.0, 0.99, 0.97, 1.02, 1.03, 0.97]
        )

        data = pd.DataFrame({
            'col1': [-0.069, -0.061, -0.086, -0.086, -0.078, 0.073, 0.057, 0.098, 0.107, 0.057],
            'col2': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        })

        # Run
        output = transformer._reverse_transform(data)

        # Asserts
        expected = pd.Series([0.01, 0.02, -0.01, -0.01, 0.0, 0.99, 0.97, 1.02, 1.03, 0.97])
        assert (output == expected).all()

        transformer._reverse_transform_helper.assert_called_once()
        call_data = np.array([
            [-0.069, -0.061, -0.086, -0.086, -0.078, 0.073, 0.057, 0.098, 0.107, 0.057],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        ]).transpose()
        np.testing.assert_allclose(
            transformer._reverse_transform_helper.call_args[0][0],
            call_data
        )

    def test__reverse_transform_nan(self):
        """Test ``_reverse_transform`` with ``np.nan`` values.

        Validate that the method correctly calls ``_reverse_transform_helper`` and produces the
        appropriate output when passed a numpy array containing ``np.nan`` values.

        Setup:
            - create an instance of the ``BayesGMMTransformer`` where the ``output_columns``
            is a list of two columns.
            - mock the `_reverse_transform_helper` with the appropriate return value.
            - set ``null_transformer`` to ``NullTransformer`` with ``null_column`` as True,
            then fit it to a pandas Series.

        Input:
            - a numpy ndarray containing transformed ``np.nan`` values.

        Ouput:
            - a pandas Series with the reverse transformed data.

        Side Effects:
            - ``_reverse_transform_helper`` should be called once with the correct data.
        """
        # Setup
        transformer = BayesGMMTransformer(max_clusters=3)
        transformer.output_columns = ['col.normalized', 'col.component']
        transformer._reverse_transform_helper = Mock()
        transformer._reverse_transform_helper.return_value = np.array([
            0.68351419, 0.67292805, 0.66234274, 0.66234274, 0.67292805,
            0.63579893, 0.62239389, 0.67292805, 0.67292805, 0.62239389
        ])

        transformer.null_transformer = NullTransformer(None, null_column=True)
        transformer.null_transformer.fit(pd.Series([0, np.nan]))

        data = np.array([
            [-0.033, -0.046, -0.058, -0.058, -0.046, 0.134, 0.122, -0.046, -0.046, 0.122],
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 1],
            [1, 1, 1, 1, 1, 0, 0, 0, 1, 0]
        ]).transpose()

        # Run
        output = transformer._reverse_transform(data)

        # Asserts
        expected = pd.Series(
            [np.nan, np.nan, np.nan, np.nan, np.nan, 0.63, 0.62, 0.67, np.nan, 0.62]
        )
        np.testing.assert_allclose(expected, output, rtol=1e-2)

        call_data = np.array([
            [-0.033385, 0., 1.],
            [-0.046177, 0., 1.],
            [-0.058968, 0., 1.],
            [-0.058968, 0., 1.],
            [-0.046177, 0., 1.],
            [0.134944, 1., 0.],
            [0.1226, 1., 0.],
            [-0.046177, 0., 0.],
            [-0.046177, 0., 1.],
            [0.1226, 1., 0.]
        ])
        transformer._reverse_transform_helper.assert_called_once()
        np.testing.assert_allclose(
            transformer._reverse_transform_helper.call_args[0][0],
            call_data,
            rtol=1e-1
        )
