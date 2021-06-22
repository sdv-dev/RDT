from unittest import TestCase

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
        should not set its ``rounding`` or ``rounding_digits`` instance
        variables.

        Input:
        - An array with floats rounded to one decimal and a None value
        Side Effect:
        - ``rounding`` and ``rounding_digits`` continue to be ``None``
        """
        # Setup
        data = np.array([1.5, None, 2.5])

        # Run
        transformer = NumericalTransformer(dtype=np.float, nan='nan', rounding=None)
        transformer.fit(data)

        # Asserts
        assert transformer.rounding is None
        assert transformer.rounding_digits is None

    def test_fit_rounding_int(self):
        """Test fit rounding parameter with int

        If the rounding parameter is set to ``None``, the ``fit`` method
        should not set its ``rounding`` or ``rounding_digits`` instance
        variables.

        Input:
        - An array with floats rounded to one decimal and a None value
        Side Effect:
        - ``rounding`` and ``rounding_digits`` are the provided int
        """
        # Setup
        data = np.array([1.5, None, 2.5])
        expected_digits = 3

        # Run
        transformer = NumericalTransformer(dtype=np.float, nan='nan', rounding=expected_digits)
        transformer.fit(data)

        # Asserts
        assert transformer.rounding == expected_digits
        assert transformer.rounding_digits == expected_digits

    def test_fit_rounding_auto(self):
        """Test fit rounding parameter with ``'auto'``

        If the ``rounding`` parameter is set to ``'auto'``,
        fit should learn the ``rounding_digits`` to be the max
        number of decimal places seen in the data.

        Input:
        - Array of floats with up to 4 decimals
        Side Effect:
        - ``rounding_digits`` is set to 4
        """
        # Setup
        data = np.array([1, 2.1, 3.12, 4.123, 5.1234, 6.123, 7.12, 8.1, 9])

        # Run
        transformer = NumericalTransformer(dtype=np.float, nan='nan', rounding='auto')
        transformer.fit(data)

        # Asserts
        assert transformer.rounding_digits == 4

    def test_fit_rounding_auto_large_numbers(self):
        """Test fit rounding parameter with ``'auto'``

        If the ``rounding`` parameter is set to ``'auto'``
        and the data is very large, fit should learn
        ``rounding_digits`` to be the biggest number of 0s
        to round to that keeps the data the same.

        Input:
        - Array of data with numbers between 10^10 and 10^20
        Side Effect:
        - ``rounding_digits`` is set to the minimum exponent seen in the data
        """
        # Setup
        exponents = [np.random.randint(10, 20) for i in range(10)]
        big_numbers = [10**exponents[i] for i in range(10)]
        data = np.array(big_numbers)

        # Run
        transformer = NumericalTransformer(dtype=np.float, nan='nan', rounding='auto')
        transformer.fit(data)

        # Asserts
        assert transformer.rounding_digits == -min(exponents)

    def test_fit_rounding_auto_max_decimals(self):
        """Test fit rounding parameter with ``'auto'``

        If the ``rounding`` parameter is set to ``'auto'``,
        fit should learn the ``rounding_digits`` to be the max
        number of decimal places seen in the data. The max
        amount of decimals that floats can be accurately compared
        with is 15. If the input data has values with more than
        14 decimals, we will not be able to accurately learn the
        number of decimal places required, so we do not round.

        Input:
        - Array with a value that has 15 decimals
        Side Effect:
        - ``rounding_digits`` is set to ``None``
        """
        # Setup
        data = np.array([0.000000000000001])

        # Run
        transformer = NumericalTransformer(dtype=np.float, nan='nan', rounding='auto')
        transformer.fit(data)

        # Asserts
        assert transformer.rounding_digits is None

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
        transformer.rounding_digits = None
        result = transformer.reverse_transform(data)

        # Assert
        np.testing.assert_array_equal(result, data)

    def test_reverse_transform_rounding_positive_rounding(self):
        """Test ``reverse_transform`` when ``rounding`` is a positive int

        The data should round to the maximum number of decimal places
        set in the ``rounding_digits`` value.

        Input:
        - Array with decimals

        Output:
        - Same array rounded to the provided number of decimal places
        """
        # Setup
        data = np.array([1.1111, 2.2222, 3.3333, 4.44444, 5.555555])

        # Run
        transformer = NumericalTransformer(dtype=np.float, nan=None)
        transformer.rounding_digits = 2
        result = transformer.reverse_transform(data)

        # Assert
        expected_data = np.array([1.11, 2.22, 3.33, 4.44, 5.56])
        np.testing.assert_array_equal(result, expected_data)

    def test_reverse_transform_rounding_negative_rounding_int(self):
        """Test ``reverse_transform`` when ``rounding`` is a negative int

        The data should round to the number set in the ``rounding_digits``
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
        transformer.rounding_digits = -3
        result = transformer.reverse_transform(data)

        # Assert
        expected_data = np.array([2000, 0, 3000, 40000])
        np.testing.assert_array_equal(result, expected_data)
        assert result.dtype == np.int

    def test_reverse_transform_rounding_negative_rounding_float(self):
        """Test ``reverse_transform`` when ``rounding`` is a negative int

        The data should round to the number set in the ``rounding_digits``
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
        transformer.rounding_digits = -3
        result = transformer.reverse_transform(data)

        # Assert
        expected_data = np.array([2000.0, 0.0, 3000.0, 40000.0])
        np.testing.assert_array_equal(result, expected_data)
        assert result.dtype == np.float

    def test_reverse_transform_rounding_zero(self):
        """Test ``reverse_transform`` when ``rounding`` is a negative int

        The data should round to the number set in the ``rounding_digits``
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
        transformer.rounding_digits = 0
        result = transformer.reverse_transform(data)

        # Assert
        expected_data = np.array([2001, 120, 3101, 4010])
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
