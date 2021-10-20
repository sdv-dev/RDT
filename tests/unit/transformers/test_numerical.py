from unittest import TestCase
from unittest.mock import Mock

import copulas
import numpy as np
import pandas as pd
import pytest

from rdt.transformers.numerical import (
    GaussianCopulaTransformer, NumericalBoundedTransformer, NumericalRoundedBoundedTransformer,
    NumericalRoundedTransformer, NumericalTransformer)


class TestNumericalTransformer(TestCase):

    def test___init__super_attrs(self):
        """super() arguments are properly passed and set as attributes."""
        nt = NumericalTransformer(dtype='int', nan='mode', null_column=False)

        assert nt.dtype == 'int'
        assert nt.nan == 'mode'
        assert nt.null_column is False

    def test__fit(self):
        """Test _fit nan mean with numpy.array"""
        # Setup
        data = pd.DataFrame([1.5, None, 2.5], columns=['a'])

        # Run
        transformer = NumericalTransformer(dtype=float, nan='nan')
        transformer._fit(data)

        # Asserts
        expect_fill_value = 'nan'
        expect_dtype = float

        assert transformer.null_transformer.fill_value == expect_fill_value
        assert transformer._dtype == expect_dtype

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

    def test__get_univariate_instance(self):
        """If a univariate instance is passed, make a copy."""
        distribution = copulas.univariate.Univariate()
        ct = GaussianCopulaTransformer(distribution=distribution)

        univariate = ct._get_univariate()

        assert univariate is not distribution
        assert isinstance(univariate, copulas.univariate.Univariate)
        assert dir(univariate) == dir(distribution)

    def test__get_univariate_tuple(self):
        """If a tuple is passed, create an instance using the given args."""
        distribution = (
            copulas.univariate.Univariate,
            {'candidates': 'a_candidates_list'}
        )
        ct = GaussianCopulaTransformer(distribution=distribution)

        univariate = ct._get_univariate()

        assert isinstance(univariate, copulas.univariate.Univariate)
        assert univariate.candidates == 'a_candidates_list'

    def test__get_univariate_class(self):
        """If a class is passed, create an instance without args."""
        distribution = copulas.univariate.Univariate
        ct = GaussianCopulaTransformer(distribution=distribution)

        univariate = ct._get_univariate()

        assert isinstance(univariate, copulas.univariate.Univariate)

    def test__get_univariate_error(self):
        """If something else is passed, rasie a TypeError."""
        distribution = 123
        ct = GaussianCopulaTransformer(distribution=distribution)

        with pytest.raises(TypeError):
            ct._get_univariate()


class TestNumericalBoundedTransformer(TestCase):

    def test___init__(self):
        """super() arguments are properly passed and set as attributes."""
        nt = NumericalBoundedTransformer(dtype='int', null_column=False)

        assert nt.dtype == 'int'
        assert nt.nan == 'mean'
        assert nt.null_column is False
        assert nt.min_value == 'auto'
        assert nt.max_value == 'auto'
        assert nt.rounding is None


class TestNumericalRoundedTransformer(TestCase):

    def test___init__(self):
        """super() arguments are properly passed and set as attributes."""
        nt = NumericalRoundedTransformer(dtype='int', null_column=False)

        assert nt.dtype == 'int'
        assert nt.nan == 'mean'
        assert nt.null_column is False
        assert nt.min_value is None
        assert nt.max_value is None
        assert nt.rounding == 'auto'


class TestNumericalRoundedBoundedTransformer(TestCase):

    def test___init__(self):
        """super() arguments are properly passed and set as attributes."""
        nt = NumericalRoundedBoundedTransformer(dtype='int', null_column=False)

        assert nt.dtype == 'int'
        assert nt.nan == 'mean'
        assert nt.null_column is False
        assert nt.min_value == 'auto'
        assert nt.max_value == 'auto'
        assert nt.rounding == 'auto'
