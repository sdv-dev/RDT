from unittest import TestCase
from unittest.mock import Mock

import copulas
import numpy as np
import pytest

from rdt.transformers.numerical import GaussianCopulaTransformer, NumericalTransformer


class TestNumericalTransformer(TestCase):

    def test___init__super_attrs(self):
        """super() arguments are properly passed and set as attributes."""
        nt = NumericalTransformer(dtype='int', nan='mode', null_column=False)

        assert nt.dtype == 'int'
        assert nt.nan == 'mode'
        assert nt.null_column is False

    def test_fit(self):
        """Test fit nan mean with numpy.array"""
        # Setup
        data = np.array([1.5, None, 2.5])

        # Run
        transformer = NumericalTransformer(dtype=np.float, nan='nan')
        transformer.fit(data)

        # Asserts
        expect_fill_value = 'nan'
        expect_dtype = np.float

        assert transformer.null_transformer.fill_value == expect_fill_value
        assert transformer._dtype == expect_dtype

    def test_fit_rounding_none(self):
        """Test fit rounding parameter with ``None``

        If the rounding parameter is set to ``None``, the ``fit`` method
        should not set its ``rounding`` or ``_rounding_digits`` instance
        variables.

        Input:
        - An array with floats rounded to one decimal and a None value
        Side Effect:
        - ``rounding`` and ``_rounding_digits`` continue to be ``None``
        """
        # Setup
        data = np.array([1.5, None, 2.5])

        # Run
        transformer = NumericalTransformer(dtype=np.float, nan='nan', rounding=None)
        transformer.fit(data)

        # Asserts
        assert transformer.rounding is None
        assert transformer._rounding_digits is None

    def test_fit_rounding_int(self):
        """Test fit rounding parameter with int

        If the rounding parameter is set to ``None``, the ``fit`` method
        should not set its ``rounding`` or ``_rounding_digits`` instance
        variables.

        Input:
        - An array with floats rounded to one decimal and a None value
        Side Effect:
        - ``rounding`` and ``_rounding_digits`` are the provided int
        """
        # Setup
        data = np.array([1.5, None, 2.5])
        expected_digits = 3

        # Run
        transformer = NumericalTransformer(dtype=np.float, nan='nan', rounding=expected_digits)
        transformer.fit(data)

        # Asserts
        assert transformer.rounding == expected_digits
        assert transformer._rounding_digits == expected_digits

    def test_fit_rounding_auto(self):
        """Test fit rounding parameter with ``'auto'``

        If the ``rounding`` parameter is set to ``'auto'``,
        ``fit`` should learn the ``_rounding_digits`` to be the max
        number of decimal places seen in the data.

        Input:
        - Array of floats with up to 4 decimals
        Side Effect:
        - ``_rounding_digits`` is set to 4
        """
        # Setup
        data = np.array([1, 2.1, 3.12, 4.123, 5.1234, 6.123, 7.12, 8.1, 9])

        # Run
        transformer = NumericalTransformer(dtype=np.float, nan='nan', rounding='auto')
        transformer.fit(data)

        # Asserts
        assert transformer._rounding_digits == 4

    def test_fit_rounding_auto_large_numbers(self):
        """Test fit rounding parameter with ``'auto'``

        If the ``rounding`` parameter is set to ``'auto'``
        and the data is very large, ``fit`` should learn
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
        data = np.array(big_numbers)

        # Run
        transformer = NumericalTransformer(dtype=np.float, nan='nan', rounding='auto')
        transformer.fit(data)

        # Asserts
        assert transformer._rounding_digits == -min(exponents)

    def test_fit_rounding_auto_max_decimals(self):
        """Test fit rounding parameter with ``'auto'``

        If the ``rounding`` parameter is set to ``'auto'``,
        ``fit`` should learn the ``_rounding_digits`` to be the max
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
        data = np.array([0.000000000000001])

        # Run
        transformer = NumericalTransformer(dtype=np.float, nan='nan', rounding='auto')
        transformer.fit(data)

        # Asserts
        assert transformer._rounding_digits is None

    def test_fit_rounding_auto_max_inf(self):
        """Test fit rounding parameter with ``'auto'``

        If the ``rounding`` parameter is set to ``'auto'``,
        and the data contains infinite values, ``fit`` should
        learn the ``_rounding_digits`` to be the min
        number of decimal places seen in the data with
        the infinite values filtered out.

        Input:
        - Array with ``np.inf`` as a value
        Side Effect:
        - ``_rounding_digits`` is set to max seen in rest of data
        """
        # Setup
        data = np.array([15000, 4000, 60000, np.inf])

        # Run
        transformer = NumericalTransformer(dtype=np.float, nan='nan', rounding='auto')
        transformer.fit(data)

        # Asserts
        assert transformer._rounding_digits is -3

    def test_fit_rounding_auto_max_zero(self):
        """Test fit rounding parameter with ``'auto'``

        If the ``rounding`` parameter is set to ``'auto'``,
        and the max in the data is 0, ``fit`` should
        learn the ``_rounding_digits`` to be 0.

        Input:
        - Array with 0 as max value
        Side Effect:
        - ``_rounding_digits`` is set to 0
        """
        # Setup
        data = np.array([0, 0, 0])

        # Run
        transformer = NumericalTransformer(dtype=np.float, nan='nan', rounding='auto')
        transformer.fit(data)

        # Asserts
        assert transformer._rounding_digits == 0

    def test_fit_rounding_auto_max_negative(self):
        """Test fit rounding parameter with ``'auto'``

        If the ``rounding`` parameter is set to ``'auto'``,
        and the max in the data is negative, the ``fit`` method
        should learn ``_rounding_digits`` to be the min number
        of digits seen in those negative values.

        Input:
        - Array with negative max value
        Side Effect:
        - ``_rounding_digits`` is set to min number of digits in array
        """
        # Setup
        data = np.array([-500, -220, -10])

        # Run
        transformer = NumericalTransformer(dtype=np.float, nan='nan', rounding='auto')
        transformer.fit(data)

        # Asserts
        assert transformer._rounding_digits == -1

    def test_fit_min_max_none(self):
        """Test fit min and max parameters with ``None``

        If the min and max parameters are set to ``None``,
        the ``fit`` method should not set its ``min`` or ``max``
        instance variables.

        Input:
        - Array of floats and null values
        Side Effect:
        - ``_min_value`` and ``_max_value`` stay ``None``
        """
        # Setup
        data = np.array([1.5, None, 2.5])

        # Run
        transformer = NumericalTransformer(dtype=np.float, nan='nan',
                                           min_value=None, max_value=None)
        transformer.fit(data)

        # Asserts
        assert transformer._min_value is None
        assert transformer._max_value is None

    def test_fit_min_max_int(self):
        """Test fit min and max parameters with int values

        If the min and max parameters are set to an int,
        the ``fit`` method should not change them.

        Input:
        - Array of floats and null values
        Side Effect:
        - ``_min_value`` and ``_max_value`` remain unchanged
        """
        # Setup
        data = np.array([1.5, None, 2.5])

        # Run
        transformer = NumericalTransformer(dtype=np.float, nan='nan',
                                           min_value=1, max_value=10)
        transformer.fit(data)

        # Asserts
        assert transformer._min_value == 1
        assert transformer._max_value == 10

    def test_fit_min_max_auto(self):
        """Test fit min and max parameters with ``'auto'``

        If the min or max parameters are set to ``'auto'``
        the ``fit`` method should learn them from the
        fitted data.

        Input:
        - Array of floats and null values
        Side Effect:
        - ``_min_value`` and ``_max_value`` are learned
        """
        # Setup
        data = np.array([-100, -5000, 0, None, 100, 4000])

        # Run
        transformer = NumericalTransformer(dtype=np.float, nan='nan',
                                           min_value='auto', max_value='auto')
        transformer.fit(data)

        # Asserts
        assert transformer._min_value == -5000
        assert transformer._max_value == 4000

    def test_reverse_transform_rounding_none(self):
        """Test ``reverse_transform`` when ``rounding`` is ``None``

        The data should not be rounded at all.

        Input:
        - Random array of floats between 0 and 1
        Output:
        - Input array
        """
        # Setup
        data = np.random.random(10)

        # Run
        transformer = NumericalTransformer(dtype=np.float, nan=None)
        transformer._rounding_digits = None
        result = transformer.reverse_transform(data)

        # Assert
        np.testing.assert_array_equal(result, data)

    def test_reverse_transform_rounding_positive_rounding(self):
        """Test ``reverse_transform`` when ``rounding`` is a positive int

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
        transformer = NumericalTransformer(dtype=np.float, nan=None)
        transformer._rounding_digits = 2
        result = transformer.reverse_transform(data)

        # Assert
        expected_data = np.array([1.11, 2.22, 3.33, 4.44, 5.56])
        np.testing.assert_array_equal(result, expected_data)

    def test_reverse_transform_rounding_negative_rounding_int(self):
        """Test ``reverse_transform`` when ``rounding`` is a negative int

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
        transformer = NumericalTransformer(dtype=np.int, nan=None)
        transformer._dtype = np.int
        transformer._rounding_digits = -3
        result = transformer.reverse_transform(data)

        # Assert
        expected_data = np.array([2000, 0, 3000, 40000])
        np.testing.assert_array_equal(result, expected_data)
        assert result.dtype == np.int

    def test_reverse_transform_rounding_negative_rounding_float(self):
        """Test ``reverse_transform`` when ``rounding`` is a negative int

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
        transformer = NumericalTransformer(dtype=np.float, nan=None)
        transformer._rounding_digits = -3
        result = transformer.reverse_transform(data)

        # Assert
        expected_data = np.array([2000.0, 0.0, 3000.0, 40000.0])
        np.testing.assert_array_equal(result, expected_data)
        assert result.dtype == np.float

    def test_reverse_transform_rounding_zero(self):
        """Test ``reverse_transform`` when ``rounding`` is a negative int

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
        transformer = NumericalTransformer(dtype=np.float, nan=None)
        transformer._rounding_digits = 0
        result = transformer.reverse_transform(data)

        # Assert
        expected_data = np.array([2001, 120, 3101, 4010])
        np.testing.assert_array_equal(result, expected_data)

    def test_reverse_transform_min_no_max(self):
        """Test reverse_transform with ``min_value`` set

        The ``reverse_transform`` method should clip any values below
        the ``min_value`` if it is set.

        Input:
        - Array with values below the min and infinitely high values
        Output:
        - Array with low values clipped to min
        """
        # Setup
        data = np.array([-np.inf, -5000, -301, -250, 0, 125, 400, np.inf])

        # Run
        transformer = NumericalTransformer(dtype=np.float, nan=None)
        transformer._min_value = -300
        result = transformer.reverse_transform(data)

        # Asserts
        expected_data = np.array([-300, -300, -300, -250, 0, 125, 400, np.inf])
        np.testing.assert_array_equal(result, expected_data)

    def test_reverse_transform_max_no_min(self):
        """Test reverse_transform with ``max_value`` set

        The ``reverse_transform`` method should clip any values above
        the ``max_value`` if it is set.

        Input:
        - Array with values above the max and infinitely low values
        Output:
        - Array with values clipped to max
        """
        # Setup
        data = np.array([-np.inf, -5000, -301, -250, 0, 125, 401, np.inf])

        # Run
        transformer = NumericalTransformer(dtype=np.float, nan=None)
        transformer._max_value = 400
        result = transformer.reverse_transform(data)

        # Asserts
        expected_data = np.array([-np.inf, -5000, -301, -250, 0, 125, 400, 400])
        np.testing.assert_array_equal(result, expected_data)

    def test_reverse_transform_min_and_max(self):
        """Test reverse_transform with ``min_value`` and ``max_value`` set

        The ``reverse_transform`` method should clip any values above
        the ``max_value`` and any values below the ``min_value``.

        Input:
        - Array with values above the max and below the min
        Output:
        - Array with out of bound values clipped to min and max
        """
        # Setup
        data = np.array([-np.inf, -5000, -301, -250, 0, 125, 401, np.inf])

        # Run
        transformer = NumericalTransformer(dtype=np.float, nan=None)
        transformer._max_value = 400
        transformer._min_value = -300
        result = transformer.reverse_transform(data)

        # Asserts
        np.testing.assert_array_equal(result, np.array([-300, -300, -300, -250, 0, 125, 400, 400]))

    def test_reverse_transform_min_an_max_with_nulls(self):
        """Test reverse_transform with nulls and ``min_value`` and ``max_value`` set

        The ``reverse_transform`` method should clip any values above
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
        transformer = NumericalTransformer(dtype=np.float, nan='nan')
        transformer._max_value = 400
        transformer._min_value = -300
        transformer.null_transformer = Mock()
        transformer.null_transformer.reverse_transform.return_value = expected_data
        result = transformer.reverse_transform(data)

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
