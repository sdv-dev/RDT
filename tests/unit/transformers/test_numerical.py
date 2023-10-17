import re
from unittest import TestCase
from unittest.mock import Mock, patch

import copulas
import numpy as np
import pandas as pd
import pytest
from copulas import univariate

from rdt.transformers.null import NullTransformer
from rdt.transformers.numerical import ClusterBasedNormalizer, FloatFormatter, GaussianNormalizer


class TestFloatFormatter(TestCase):

    def test___init__super_attrs(self):
        """super() arguments are properly passed and set as attributes."""
        nt = FloatFormatter(
            missing_value_replacement='mode',
            missing_value_generation='random'
        )

        assert nt.missing_value_replacement == 'mode'
        assert nt.missing_value_generation == 'random'

    @patch('rdt.transformers.numerical.LOGGER')
    def test__learn_rounding_digits_more_than_15_decimals(self, logger_mock):
        """Test the _learn_rounding_digits method with more than 15 decimals.

        If the data has more than 15 decimals, return None and raise warning.
        """
        # Setup
        data = pd.Series(np.random.random(size=10).round(20), name='col')

        # Run
        output = FloatFormatter._learn_rounding_digits(data)

        # Assert
        logger_msg = "No rounding scheme detected for column '%s'. Data will not be rounded."
        logger_mock.info.assert_called_once_with(logger_msg, 'col')
        assert output is None

    def test__learn_rounding_digits_less_than_15_decimals(self):
        """Test the _learn_rounding_digits method with less than 15 decimals.

        If the data has less than 15 decimals, the maximum number of decimals
        should be returned.

        Input:
        - An array that contains floats with a maximum of 3 decimals and a
          NaN.
        Output:
        - 3
        """
        data = pd.Series(np.array([10, 0., 0.1, 0.12, 0.123, np.nan]))

        output = FloatFormatter._learn_rounding_digits(data)

        assert output == 3

    def test__learn_rounding_digits_negative_decimals_float(self):
        """Test the _learn_rounding_digits method with floats multiples of powers of 10.

        If the data has all multiples of 10 the output should be 0.

        Input:
        - An array that contains floats that are multiples of powers of 10, 100 and 1000 and a NaN.
        """
        data = pd.Series(np.array([1230., 12300., 123000., np.nan]))

        output = FloatFormatter._learn_rounding_digits(data)

        assert output == 0

    def test__learn_rounding_digits_negative_decimals_integer(self):
        """Test the _learn_rounding_digits method with integers multiples of powers of 10.

        If the data has all multiples of 10 the output should be 0.

        Input:
        - An array that contains integers that are multiples of powers of 10, 100 and 1000
          and a NaN.
        """
        data = pd.Series(np.array([1230, 12300, 123000, np.nan]))

        output = FloatFormatter._learn_rounding_digits(data)

        assert output == 0

    def test__learn_rounding_digits_all_missing_value_replacements(self):
        """Test the _learn_rounding_digits method with data that is all NaNs.

        If the data is all NaNs, expect that the output is None.

        Input:
        - An array of NaN.
        Output:
        - None
        """
        data = pd.Series(np.array([np.nan, np.nan, np.nan, np.nan]))

        output = FloatFormatter._learn_rounding_digits(data)

        assert output is None

    def test__validate_values_within_bounds(self):
        """Test the ``_validate_values_within_bounds`` method.

        If all values are correctly bounded, it shouldn't do anything.

        Setup:
            - instantiate ``FloatFormatter`` with ``computer_representation`` set to an int.
        Input:
            - a Dataframe.
        """
        # Setup
        data = pd.Series([15, None, 25])
        transformer = FloatFormatter()
        transformer.computer_representation = 'UInt8'

        # Run
        transformer._validate_values_within_bounds(data)

    def test__validate_values_within_bounds_under_minimum(self):
        """Test the ``_validate_values_within_bounds`` method.

        Expected to crash if a value is under the bound.

        Setup:
            - instantiate ``FloatFormatter`` with ``computer_representation`` set to an int.
        Input:
            - a Dataframe.
        Side Effect:
            - raise ``ValueError``.
        """
        # Setup
        data = pd.Series([-15, None, 0], name='a')
        transformer = FloatFormatter()
        transformer.computer_representation = 'UInt8'

        # Run / Assert
        err_msg = re.escape(
            "The minimum value in column 'a' is -15.0. All values represented by 'UInt8'"
            ' must be in the range [0, 255].'
        )
        with pytest.raises(ValueError, match=err_msg):
            transformer._validate_values_within_bounds(data)

    def test__validate_values_within_bounds_over_maximum(self):
        """Test the ``_validate_values_within_bounds`` method.

        Expected to crash if a value is over the bound.

        Setup:
            - instantiate ``FloatFormatter`` with ``computer_representation`` set to an int.
        Input:
            - a Dataframe.
        """
        # Setup
        data = pd.Series([255, None, 256], name='a')
        transformer = FloatFormatter()
        transformer.computer_representation = 'UInt8'

        # Run / Assert
        err_msg = re.escape(
            "The maximum value in column 'a' is 256.0. All values represented by 'UInt8'"
            ' must be in the range [0, 255].'
        )
        with pytest.raises(ValueError, match=err_msg):
            transformer._validate_values_within_bounds(data)

    def test__validate_values_within_bounds_floats(self):
        """Test the ``_validate_values_within_bounds`` method.

        Expected to crash if float values are passed when ``computer_representation`` is an int.

        Setup:
            - instantiate ``FloatFormatter`` with ``computer_representation`` set to an int.
        Input:
            - a Dataframe.
        """
        # Setup
        data = pd.Series([249.2, None, 250.0, 10.2], name='a')
        transformer = FloatFormatter()
        transformer.computer_representation = 'UInt8'

        # Run / Assert
        err_msg = re.escape(
            "The column 'a' contains float values [249.2, 10.2]."
            " All values represented by 'UInt8' must be integers."
        )
        with pytest.raises(ValueError, match=err_msg):
            transformer._validate_values_within_bounds(data)

    def test__fit(self):
        """Test the ``_fit`` method.

        Validate that the ``_dtype`` and ``.null_transformer.missing_value_replacement`` attributes
        are set correctly.

        Setup:
            - initialize a ``FloatFormatter`` with the ``missing_value_replacement``
              parameter set to ``'missing_value_replacement'``.

        Input:
            - a pandas series containing a None.

        Side effect:
            - it sets the ``null_transformer.missing_value_replacement``.
            - it sets the ``_dtype``.
            - it calls ``_validate_values_within_bounds``.
        """
        # Setup
        data = pd.Series([1.5, None, 2.5])
        transformer = FloatFormatter(
            missing_value_replacement='missing_value_replacement'
        )
        transformer._validate_values_within_bounds = Mock()

        # Run
        transformer._fit(data)

        # Asserts
        expected = 'missing_value_replacement'
        assert transformer.null_transformer._missing_value_replacement == expected
        assert transformer._dtype == float
        assert transformer.output_properties == {
            None: {'sdtype': 'float', 'next_transformer': None}
        }
        transformer._validate_values_within_bounds.assert_called_once_with(data)
        assert transformer.output_properties == {
            None: {'sdtype': 'float', 'next_transformer': None},
        }

    def test__fit_learn_rounding_scheme_false(self):
        """Test ``_fit`` with ``learn_rounding_scheme`` set to ``False``.

        If the ``learn_rounding_scheme`` is set to ``False``, the ``_fit`` method
        should not set its ``_rounding_digits`` instance variable.

        Input:
        - An array with floats rounded to one decimal and a None value
        Side Effect:
        - ``_rounding_digits`` should be ``None``
        """
        # Setup
        data = pd.Series([1.5, None, 2.5])

        # Run
        transformer = FloatFormatter(
            missing_value_replacement='missing_value_replacement',
            learn_rounding_scheme=False
        )
        transformer._fit(data)

        # Asserts
        assert transformer._rounding_digits is None

    def test__fit_learn_rounding_scheme_true(self):
        """Test ``_fit`` with ``learn_rounding_scheme`` set to ``True``.

        If ``learn_rounding_scheme`` is set to ``True``, the ``_fit`` method
        should set its ``_rounding_digits`` instance variable to what is learned
        in the data.

        Input:
        - A Series with floats up to 4 decimals and a None value
        Side Effect:
        - ``_rounding_digits`` is set to 4
        """
        # Setup
        data = pd.Series([1, 2.1, 3.12, 4.123, 5.1234, 6.123, 7.12, 8.1, 9, None])

        # Run
        transformer = FloatFormatter(
            missing_value_replacement='mean',
            learn_rounding_scheme=True
        )
        transformer._fit(data)

        # Asserts
        assert transformer._rounding_digits == 4

    def test__fit_learn_rounding_scheme_true_max_decimals(self):
        """Test ``_fit`` with ``learn_rounding_scheme`` set to ``True``.

        If the ``learn_rounding_scheme`` parameter is set to ``True``, ``_fit`` should learn
        the ``_rounding_digits`` to be the max number of decimal places seen in the data.
        The max amount of decimals that floats can be accurately compared with is 15.
        If the input data has values with more than 14 decimals, we will not be able to
        accurately learn the number of decimal places required, so we do not round.

        Input:
        - Series with a value that has 15 decimals
        Side Effect:
        - ``_rounding_digits`` is set to None
        """
        # Setup
        data = pd.Series([0.000000000000001])

        # Run
        transformer = FloatFormatter(
            missing_value_replacement='mean',
            learn_rounding_scheme=True
        )
        transformer._fit(data)

        # Asserts
        assert transformer._rounding_digits is None

    def test__fit_learn_rounding_scheme_true_inf(self):
        """Test ``_fit`` with ``learn_rounding_scheme`` set to ``True``.

        If the ``learn_rounding_scheme`` parameter is set to ``True``, and the data
        contains only integers or infinite values, ``_fit`` should learn
        ``_rounding_digits`` to be 0.


        Input:
        - Series with ``np.inf`` as a value
        Side Effect:
        - ``_rounding_digits`` is set to 0
        """
        # Setup
        data = pd.Series([15000, 4000, 60000, np.inf])

        # Run
        transformer = FloatFormatter(
            missing_value_replacement='mean',
            learn_rounding_scheme=True
        )
        transformer._fit(data)

        # Asserts
        assert transformer._rounding_digits == 0

    def test__fit_learn_rounding_scheme_true_max_zero(self):
        """Test ``_fit`` with ``learn_rounding_scheme`` set to ``True``.

        If the ``learn_rounding_scheme`` parameter is set to ``True``, and the max
        in the data is 0, ``_fit`` should learn the ``_rounding_digits`` to be 0.

        Input:
        - Series with 0 as max value
        Side Effect:
        - ``_rounding_digits`` is set to 0
        """
        # Setup
        data = pd.Series([0, 0, 0])

        # Run
        transformer = FloatFormatter(
            missing_value_replacement='mean',
            learn_rounding_scheme=True
        )
        transformer._fit(data)

        # Asserts
        assert transformer._rounding_digits == 0

    def test__fit_enforce_min_max_values_false(self):
        """Test ``_fit`` with ``enforce_min_max_values`` set to ``False``.

        If the ``enforce_min_max_values`` parameter is set to ``False``,
        the ``_fit`` method should not set its ``min`` or ``max``
        instance variables.

        Input:
        - Series of floats and null values
        Side Effect:
        - ``_min_value`` and ``_max_value`` stay ``None``
        """
        # Setup
        data = pd.Series([1.5, None, 2.5])

        # Run
        transformer = FloatFormatter(
            missing_value_replacement='mean',
            enforce_min_max_values=False
        )
        transformer._fit(data)

        # Asserts
        assert transformer._min_value is None
        assert transformer._max_value is None

    def test__fit_enforce_min_max_values_true(self):
        """Test ``_fit`` with ``enforce_min_max_values`` set to ``True``.

        If the ``enforce_min_max_values`` parameter is set to ``True``,
        the ``_fit`` method should learn the min and max values from the _fitted data.

        Input:
        - Series of floats and null values
        Side Effect:
        - ``_min_value`` and ``_max_value`` are learned
        """
        # Setup
        data = pd.Series([-100, -5000, 0, None, 100, 4000])

        # Run
        transformer = FloatFormatter(
            missing_value_replacement='mean',
            enforce_min_max_values=True
        )
        transformer._fit(data)

        # Asserts
        assert transformer._min_value == -5000
        assert transformer._max_value == 4000

    def test__fit_missing_value_replacement_from_column(self):
        """Test output_properties contains 'is_null' column.

        When ``missing_value_generation`` is ``from_column`` an output property ``is_null`` should
        exist.
        """
        # Setup
        transformer = FloatFormatter(missing_value_generation='from_column')
        data = pd.Series([1, np.nan])

        # Run
        transformer._fit(data)

        # Assert
        assert transformer.output_properties == {
            None: {'sdtype': 'float', 'next_transformer': None},
            'is_null': {'sdtype': 'float', 'next_transformer': None},
        }

    def test__transform(self):
        """Test the ``_transform`` method.

        Validate that this method calls the ``self.null_transformer.transform`` method once.

        Setup:
            - create an instance of a ``FloatFormatter`` and set ``self.null_transformer``
            to a ``NullTransformer``.

        Input:
            - a pandas series.

        Output:
            - the transformed numpy array.
        """
        # Setup
        data = pd.Series([1, 2, 3])
        transformer = FloatFormatter()
        transformer._validate_values_within_bounds = Mock()
        transformer.null_transformer = Mock()

        # Run
        transformer._transform(data)

        # Assert
        transformer._validate_values_within_bounds.assert_called_once_with(data)
        assert transformer.null_transformer.transform.call_count == 1

    def test__reverse_transform_learn_rounding_scheme_false(self):
        """Test ``_reverse_transform`` when ``learn_rounding_scheme`` is ``False``.

        The data should not be rounded at all.

        Input:
        - Random array of floats between 0 and 1
        Output:
        - Input array
        """
        # Setup
        data = np.random.random(10)

        # Run
        transformer = FloatFormatter()
        transformer.learn_rounding_scheme = False
        transformer._rounding_digits = None
        transformer.null_transformer = NullTransformer('mean')
        result = transformer._reverse_transform(data)

        # Assert
        np.testing.assert_array_equal(result, data)

    def test__reverse_transform_rounding_none_dtype_int(self):
        """Test ``_reverse_transform`` with ``_dtype`` as ``np.int64`` and no rounding.

        The data should be rounded to 0 decimals and returned as integer values if the ``_dtype``
        is ``np.int64`` even if ``_rounding_digits`` is ``None``.

        Input:
        - Array of multiple float values with decimals.
        Output:
        - Input array rounded an converted to integers.
        """
        # Setup
        data = np.array([0., 1.2, 3.45, 6.789])

        # Run
        transformer = FloatFormatter()
        transformer._rounding_digits = None
        transformer._dtype = np.int64
        transformer.null_transformer = NullTransformer('mean')
        result = transformer._reverse_transform(data)

        # Assert
        expected = np.array([0, 1, 3, 7])
        np.testing.assert_array_equal(result, expected)

    def test__reverse_transform_rounding_none_with_nulls(self):
        """Test ``_reverse_transform`` when ``_rounding_digits`` is ``None`` and there are nulls.

        The data should not be rounded at all.

        Input:
        - 2d Array of multiple float values with decimals and a column setting at least 1 null.
        Output:
        - First column of the input array as entered, replacing the indicated value with a
          missing_value_replacement.
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
        transformer = FloatFormatter()
        null_transformer = Mock()
        null_transformer.reverse_transform.return_value = np.array([0., 1.2, np.nan, 6.789])
        transformer.null_transformer = null_transformer
        transformer.learn_rounding_scheme = False
        transformer._rounding_digits = None
        transformer._dtype = float
        result = transformer._reverse_transform(data)

        # Assert
        expected = np.array([0., 1.2, np.nan, 6.789])
        np.testing.assert_array_equal(result, expected)

    def test__reverse_transform_rounding_none_with_nulls_dtype_int(self):
        """Test ``_reverse_transform`` rounding when dtype is int and there are nulls.

        The data should be rounded to 0 decimals and returned as float values with
        nulls in the right place.

        Input:
        - 2d Array of multiple float values with decimals and a column setting at least 1 null.
        Output:
        - First column of the input array rounded, replacing the indicated value with a
          ``NaN``, and kept as float values.
        """
        # Setup
        data = np.array([
            [0., 0.],
            [1.2, 0.],
            [3.45, 1.],
            [6.789, 0.],
        ])

        # Run
        transformer = FloatFormatter()
        null_transformer = Mock()
        null_transformer.reverse_transform.return_value = np.array([0., 1.2, np.nan, 6.789])
        transformer.null_transformer = null_transformer
        transformer.learn_rounding_digits = False
        transformer._rounding_digits = None
        transformer._dtype = int
        result = transformer._reverse_transform(data)

        # Assert
        expected = np.array([0., 1., np.nan, 7.])
        np.testing.assert_array_equal(result, expected)

    def test__reverse_transform_rounding_small_numbers(self):
        """Test ``_reverse_transform`` when ``_rounding_digits`` is positive.

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
        transformer = FloatFormatter()
        transformer.learn_rounding_scheme = True
        transformer._rounding_digits = 2
        transformer.null_transformer = NullTransformer('mean')
        result = transformer._reverse_transform(data)

        # Assert
        expected_data = np.array([1.11, 2.22, 3.33, 4.44, 5.56])
        np.testing.assert_array_equal(result, expected_data)

    def test__reverse_transform_rounding_big_numbers_type_int(self):
        """Test ``_reverse_transform`` when ``_rounding_digits`` is negative.

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
        transformer = FloatFormatter()
        transformer._dtype = int
        transformer.learn_rounding_scheme = True
        transformer._rounding_digits = -3
        transformer.null_transformer = NullTransformer('mean')
        result = transformer._reverse_transform(data)

        # Assert
        expected_data = np.array([2000, 0, 3000, 40000])
        np.testing.assert_array_equal(result, expected_data)
        assert result.dtype == int

    def test__reverse_transform_rounding_negative_type_float(self):
        """Test ``_reverse_transform`` when ``_rounding_digits`` is negative.

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
        transformer = FloatFormatter()
        transformer.learn_rounding_scheme = True
        transformer._rounding_digits = -3
        transformer.null_transformer = NullTransformer('mean')
        result = transformer._reverse_transform(data)

        # Assert
        expected_data = np.array([2000.0, 0.0, 3000.0, 40000.0])
        np.testing.assert_array_equal(result, expected_data)
        assert result.dtype == float

    def test__reverse_transform_rounding_zero_decimal_places(self):
        """Test ``_reverse_transform`` when ``_rounding_digits`` is 0.

        The data should round to the number set in the ``_rounding_digits``
        attribute.

        Input:
        - Array with with larger numbers

        Output:
        - Same array rounded to the 0s place
        """
        # Setup
        data = np.array([2000.554, 120.2, 3101, 4010])

        # Run
        transformer = FloatFormatter()
        transformer.learn_rounding_scheme = True
        transformer._rounding_digits = 0
        transformer.null_transformer = NullTransformer('mean')
        result = transformer._reverse_transform(data)

        # Assert
        expected_data = np.array([2001, 120, 3101, 4010])
        np.testing.assert_array_equal(result, expected_data)

    def test__reverse_transform_enforce_min_max_values(self):
        """Test ``_reverse_transform`` with ``enforce_min_max_values`` set to ``True``.

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
        transformer = FloatFormatter()
        transformer.enforce_min_max_values = True
        transformer._max_value = 400
        transformer._min_value = -300
        transformer.null_transformer = NullTransformer('mean')
        result = transformer._reverse_transform(data)

        # Asserts
        np.testing.assert_array_equal(result, np.array([-300, -300, -300, -250, 0, 125, 400, 400]))

    def test__reverse_transform_enforce_min_max_values_with_nulls(self):
        """Test ``_reverse_transform`` with nulls and ``enforce_min_max_values`` set to ``True``.

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
        expected_data = np.array([-300, -300, np.nan, -250, 0, np.nan, 400, 400])

        # Run
        transformer = FloatFormatter(missing_value_replacement='mean')
        transformer._max_value = 400
        transformer._min_value = -300
        transformer.enforce_min_max_values = True
        transformer.null_transformer = Mock()
        transformer.null_transformer.reverse_transform.return_value = expected_data
        result = transformer._reverse_transform(data)

        # Asserts
        null_transformer_calls = transformer.null_transformer.reverse_transform.mock_calls
        np.testing.assert_array_equal(null_transformer_calls[0][1][0], data)
        np.testing.assert_array_equal(result, expected_data)

    def test__reverse_transform_enforce_computer_representation(self):
        """Test ``_reverse_transform`` with ``computer_representation`` set to ``Int8``.

        The ``_reverse_transform`` method should clip any values out of bounds.

        Input:
        - Array with values above the max and below the min
        Output:
        - Array with out of bound values clipped to min and max
        """
        # Setup
        data = np.array([-np.inf, np.nan, -5000, -301, -100, 0, 125, 401, np.inf])

        # Run
        transformer = FloatFormatter(computer_representation='Int8')
        transformer.null_transformer = NullTransformer('mean')
        result = transformer._reverse_transform(data)

        # Asserts
        np.testing.assert_array_equal(
            result, np.array([-128, np.nan, -128, -128, -100, 0, 125, 127, 127])
        )


class TestGaussianNormalizer:

    def test___init__super_attrs(self):
        """super() arguments are properly passed and set as attributes."""
        ct = GaussianNormalizer(
            missing_value_generation='random',
            learn_rounding_scheme=False,
            enforce_min_max_values=False
        )

        assert ct.missing_value_replacement == 'mean'
        assert ct.missing_value_generation == 'random'
        assert ct.learn_rounding_scheme is False
        assert ct.enforce_min_max_values is False

    def test___init__str_distr(self):
        """If distribution is a str, it is resolved using the _DISTRIBUTIONS dict."""
        ct = GaussianNormalizer(distribution='gamma')

        assert ct._distribution is copulas.univariate.GammaUnivariate

    def test___init__non_distr(self):
        """If distribution is not an str, it is store as given."""
        univariate = copulas.univariate.Univariate()
        ct = GaussianNormalizer(distribution=univariate)

        assert ct._distribution is univariate

    def test___init__deprecated_distributions_warning(self):
        """Test it warns when using deprecated distributions."""
        # Run and Assert
        dists = zip(['gaussian', 'student_t', 'truncated_gaussian'], ['norm', 't', 'truncnorm'])
        for deprecated, distribution in dists:
            err_msg = re.escape(
                f"Future versions of RDT will not support '{deprecated}' as an option. "
                f"Please use '{distribution}' instead."
            )
            with pytest.warns(FutureWarning, match=err_msg):
                GaussianNormalizer(distribution=deprecated)

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
                GaussianNormalizer._get_distributions()

    def test__get_distributions(self):
        """Test the ``_get_distributions`` method.

        Validate that this method returns the correct dictionary of distributions.

        Setup:
            - instantiate a ``GaussianNormalizer``.
        """
        # Setup
        transformer = GaussianNormalizer()

        # Run
        distributions = transformer._get_distributions()

        # Assert
        expected = {
            'gamma': univariate.GammaUnivariate,
            'beta': univariate.BetaUnivariate,
            'gaussian_kde': univariate.GaussianKDE,
            'uniform': univariate.UniformUnivariate,
            'truncnorm': univariate.TruncatedGaussian,
            'norm': univariate.GaussianUnivariate,
            't': univariate.StudentTUnivariate,
        }
        assert distributions == expected

    def test__get_univariate_instance(self):
        """Test the ``_get_univariate`` method when the distribution is univariate.

        Validate that a deepcopy of the distribution stored in ``self._distribution`` is returned.

        Setup:
            - create an instance of a ``GaussianNormalizer`` with ``distribution`` set
            to ``univariate.Univariate``.

        Output:
            - a copy of the value stored in ``self._distribution``.
        """
        # Setup
        distribution = copulas.univariate.BetaUnivariate()
        ct = GaussianNormalizer(distribution=distribution)

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
            - create an instance of a ``GaussianNormalizer`` and set
            ``distribution`` to a tuple.

        Output:
            - an instance of ``copulas.univariate.Univariate`` with the passed arguments.
        """
        # Setup
        distribution = (
            copulas.univariate.Univariate,
            {'candidates': 'a_candidates_list'}
        )
        ct = GaussianNormalizer(distribution=distribution)

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
            - create an instance of a ``GaussianNormalizer`` and set ``distribution``
            to ``univariate.Univariate``.

        Output:
            - an instance of ``copulas.univariate.Univariate`` without any arguments.
        """
        # Setup
        distribution = copulas.univariate.BetaUnivariate
        ct = GaussianNormalizer(distribution=distribution)

        # Run
        univariate = ct._get_univariate()

        # Assert
        assert isinstance(univariate, copulas.univariate.Univariate)

    def test__get_univariate_error(self):
        """Test the ``_get_univariate`` method when ``distribution`` is invalid.

        Validate that it raises an error if an invalid distribution is stored in
        ``distribution``.

        Setup:
            - create an instance of a ``GaussianNormalizer`` and set ``self._distribution``
            improperly.

        Raise:
            - TypeError(f'Invalid distribution: {distribution}')
        """
        # Setup
        distribution = 123
        ct = GaussianNormalizer(distribution=distribution)

        # Run / Assert
        with pytest.raises(TypeError):
            ct._get_univariate()

    def test__fit(self):
        """Test the ``_fit`` method.

        Validate that ``_fit`` calls ``_get_univariate``.

        Setup:
            - create an instance of the ``GaussianNormalizer``.
            - mock the  ``_get_univariate`` method.

        Input:
            - a pandas series of float values.

        Side effect:
            - call the `_get_univariate`` method.
        """
        # Setup
        data = pd.Series([0.0, np.nan, 1.0])
        ct = GaussianNormalizer()
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
        assert ct.output_properties == {
            None: {'sdtype': 'float', 'next_transformer': None},
        }

    def test__fit_missing_value_generation_from_column(self):
        """Test the ``_fit`` method.

        Validate that ``_fit`` calls ``_get_univariate``.
        """
        # Setup
        data = pd.Series([0.0, np.nan, 1.0])
        ct = GaussianNormalizer(missing_value_generation='from_column')
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
        assert ct.output_properties == {
            None: {'sdtype': 'float', 'next_transformer': None},
            'is_null': {'sdtype': 'float', 'next_transformer': None},
        }

    @patch('rdt.transformers.numerical.warnings')
    def test__fit_catch_warnings(self, mock_warnings):
        """Test the ``_fit`` method.

        Validate that ``_fit`` uses ``catch_warnings`` and ``warnings.simplefilter``.

        Setup:
            - create an instance of the ``GaussianNormalizer``.
            - mock the  ``warnings`` package.

        Input:
            - a pandas series of float values.

        Side effect:
            - call the `warnings.catch_warnings`` method.
            - call the `warnings.simplefilter`` method.
        """
        # Setup
        data = pd.Series([0.0, 0.5, 1.0])
        ct = GaussianNormalizer()
        ct._get_univariate = Mock()

        # Run
        ct._fit(data)

        # Assert
        mock_warnings.catch_warnings.assert_called_once()
        mock_warnings.simplefilter.assert_called_once_with('ignore')

    def test__copula_transform(self):
        """Test the ``_copula_transform`` method.

        Validate that ``_copula_transform`` calls ``_get_univariate``.

        Setup:
            - create an instance of the ``GaussianNormalizer``.
            - mock  ``_univariate``.

        Input:
            - a pandas series of float values.

        Ouput:
            - a numpy array of the transformed data.
        """
        # Setup
        ct = GaussianNormalizer()
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

        Validate that ``_transform`` produces the correct values when ``missing_value_generation``
        is 'from_column'.
        """
        # Setup
        data = pd.Series([0.0, 1.0, 2.0, np.nan])
        ct = GaussianNormalizer()
        ct._univariate = Mock()
        ct._univariate.cdf.return_value = np.array([0.25, 0.5, 0.75, 0.5])
        ct.null_transformer = NullTransformer('mean', missing_value_generation='from_column')
        ct.null_transformer.fit(data)

        # Run
        transformed_data = ct._transform(data)

        # Assert
        expected = np.array([
            [-0.67449, 0, 0.67449, 0],
            [0, 0, 0, 1.0]
        ]).T
        np.testing.assert_allclose(transformed_data, expected, rtol=1e-2)

    def test__transform_missing_value_generation_is_random(self):
        """Test the ``_transform`` method.

        Validate that ``_transform`` produces the correct values when ``missing_value_generation``
        is ``random``.
        """
        # Setup
        data = pd.Series([
            0.0, 1.0, 2.0, 1.0
        ])
        ct = GaussianNormalizer()
        ct._univariate = Mock()
        ct._univariate.cdf.return_value = np.array([0.25, 0.5, 0.75, 0.5])
        ct.null_transformer = NullTransformer('mean', missing_value_generation='random')

        # Run
        ct.null_transformer.fit(data)
        transformed_data = ct._transform(data)

        # Assert
        expected = np.array([-0.67449, 0, 0.67449, 0]).T
        np.testing.assert_allclose(transformed_data, expected, rtol=1e-3)

    def test__reverse_transform(self):
        """Test the ``_reverse_transform`` method.

        Validate that ``_reverse_transform`` produces the correct values when
        ``missing_value_generation`` is 'from_column'.
        """
        # Setup
        data = np.array([
            [-0.67449, 0, 0.67449, 0],
            [0, 0, 0, 1.0],
        ]).T
        expected = pd.Series([
            0.0, 1.0, 2.0, np.nan
        ])
        ct = GaussianNormalizer()
        ct._univariate = Mock()
        ct._univariate.ppf.return_value = np.array([0.0, 1.0, 2.0, 1.0])
        ct.null_transformer = NullTransformer(
            missing_value_replacement='mean',
            missing_value_generation='from_column'
        )

        # Run
        ct.null_transformer.fit(expected)
        transformed_data = ct._reverse_transform(data)

        # Assert
        np.testing.assert_allclose(transformed_data, expected, rtol=1e-3)

    def test__reverse_transform_missing_value_generation(self):
        """Test the ``_reverse_transform`` method.

        Validate that ``_reverse_transform`` produces the correct values when
        ``missing_value_generation`` is 'random'.
        """
        # Setup
        data = pd.Series(
            [-0.67449, 0, 0.67449, 0]
        ).T
        expected = pd.Series([
            0.0, 1.0, 2.0, 1.0
        ])
        ct = GaussianNormalizer()
        ct._univariate = Mock()
        ct._univariate.ppf.return_value = np.array([0.0, 1.0, 2.0, 1.0])
        ct.null_transformer = NullTransformer(None, missing_value_generation='random')

        # Run
        ct.null_transformer.fit(expected)
        transformed_data = ct._reverse_transform(data)

        # Assert
        np.testing.assert_allclose(transformed_data, expected, rtol=1e-3)


class TestClusterBasedNormalizer(TestCase):

    def test__get_current_random_seed_random_states_is_none(self):
        """Test that the method returns 0 if ``instance.random_states`` is None."""
        # Setup
        transformer = ClusterBasedNormalizer(max_clusters=10, weight_threshold=0.005)
        transformer.random_states = None

        # Run
        random_seed = transformer._get_current_random_seed()

        # Assert
        assert random_seed == 0

    @patch('rdt.transformers.numerical.BayesianGaussianMixture')
    def test__fit(self, mock_bgm):
        """Test ``_fit``.

        Validate that the method sets the internal variables to the correct values
        when given a pandas Series.

        Setup:
            - patch a ``BayesianGaussianMixture`` with ``weights_`` containing two components
            greater than the ``weight_threshold`` parameter.
            - create an instance of the ``ClusterBasedNormalizer``.

        Input:
            - a pandas Series containing random values.

        Side Effects:
            - the sum of ``valid_component_indicator`` should equal to 2
            (the number of ``weights_`` greater than the threshold).
        """
        # Setup
        bgm_instance = mock_bgm.return_value
        bgm_instance.weights_ = np.array([10.0, 5.0, 0.0])
        transformer = ClusterBasedNormalizer(max_clusters=10, weight_threshold=0.005)
        mock_state = Mock()
        transformer.random_states['fit'] = mock_state
        mock_state.get_state.return_value = [None, [0]]
        data = pd.Series(np.random.random(size=100))

        # Run
        transformer._fit(data)

        # Asserts
        assert transformer._bgm_transformer == bgm_instance
        assert transformer.valid_component_indicator.sum() == 2
        assert transformer.output_properties == {
            'normalized': {'sdtype': 'float', 'next_transformer': None},
            'component': {'sdtype': 'categorical', 'next_transformer': None}
        }
        mock_bgm.assert_called_once_with(
            n_components=10,
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=0.001,
            random_state=0
        )

    @patch('rdt.transformers.numerical.BayesianGaussianMixture')
    def test__fit_missing_value_replacement(self, mock_bgm):
        """Test ``_fit`` with ``np.nan`` values.

        Validate that the method sets the internal variables to the correct values
        when given a pandas Series containing ``np.nan`` values.
        """
        # Setup
        bgm_instance = mock_bgm.return_value
        bgm_instance.weights_ = np.array([10.0, 5.0, 0.0])
        transformer = ClusterBasedNormalizer(
            max_clusters=10,
            weight_threshold=0.005,
            missing_value_generation='from_column'
        )

        data = pd.Series(np.random.random(size=100))
        mask = np.random.choice([1, 0], data.shape, p=[.1, .9]).astype(bool)
        data[mask] = np.nan

        # Run
        transformer._fit(data)

        # Asserts
        assert transformer._bgm_transformer == bgm_instance
        assert transformer.valid_component_indicator.sum() == 2
        assert transformer.null_transformer.models_missing_values()
        assert transformer.output_properties == {
            'normalized': {'sdtype': 'float', 'next_transformer': None},
            'component': {'sdtype': 'categorical', 'next_transformer': None},
            'is_null': {'sdtype': 'float', 'next_transformer': None},
        }

    @patch('rdt.transformers.numerical.BayesianGaussianMixture')
    @patch('rdt.transformers.numerical.warnings')
    def test__fit_catch_warnings(self, mock_warnings, mock_bgm):
        """Test ``_fit`` with ``np.nan`` values.

        Validate that ``_fit`` uses ``catch_warnings`` and ``warnings.simplefilter``.

        Setup:
            - create an instance of the ``ClusterBasedNormalizer``.
            - mock the  ``warnings`` package.

        Input:
            - a pandas series of float values.

        Side effect:
            - call the `warnings.catch_warnings`` method.
            - call the `warnings.simplefilter`` method.
        """
        # Setup
        bgm_instance = mock_bgm.return_value
        bgm_instance.weights_ = np.array([10.0, 5.0, 0.0])
        transformer = ClusterBasedNormalizer(max_clusters=10, weight_threshold=0.005)
        data = pd.Series(np.random.random(size=100))

        # Run
        transformer._fit(data)

        # Assert
        mock_warnings.catch_warnings.assert_called_once()
        mock_warnings.simplefilter.assert_called_once_with('ignore')

    def test__transform(self):
        """Test ``_transform``.

        Validate that the method produces the appropriate output when given a pandas Series.

        Setup:
            - create an instance of the ``ClusterBasedNormalizer`` where:
                - ``_bgm_transformer`` is mocked with the appropriate ``means_``, ``covariances_``
                and ``predict_proba.return_value``.
                - ``valid_component_indicator`` is set to ``np.array([True, True, False])``.

        Input:
            - a pandas Series.

        Ouput:
            - a numpy array with the transformed data.
        """
        # Setup
        random_state = np.random.get_state()
        np.random.set_state(np.random.RandomState(10).get_state())
        transformer = ClusterBasedNormalizer(max_clusters=3)
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

        transformer.valid_component_indicator = np.array([True, True, False])
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
        np.random.set_state(random_state)

    def test__transform_missing_value_replacement(self):
        """Test ``_transform`` with ``np.nan`` values.

        Validate that the method produces the appropriate output when given a pandas Series
        containing ``np.nan`` values.

        Setup:
            - create an instance of the ``ClusterBasedNormalizer`` where:
                - ``_bgm_transformer`` is mocked with the appropriate ``means_``, ``covariances_``
                and ``predict_proba.return_value``.
                - ``valid_component_indicator`` is set to ``np.array([True, True, False])``.

        Input:
            - a pandas Series.

        Ouput:
            - a numpy array with the transformed data.
        """
        # Setup
        random_state = np.random.get_state()
        np.random.set_state(np.random.RandomState(10).get_state())
        transformer = ClusterBasedNormalizer(max_clusters=3)
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

        transformer.valid_component_indicator = np.array([True, True, False])
        transformer.null_transformer = NullTransformer(0.0, missing_value_generation='from_column')
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
        np.random.set_state(random_state)

    def test__reverse_transform_helper(self):
        """Test ``_reverse_transform_helper``.

        Validate that the method produces the appropriate output when passed a numpy array.

        Setup:
            - create an instance of the ``ClusterBasedNormalizer`` where:
                - ``_bgm_transformer`` is mocked with the appropriate
                ``means_`` and ``covariances_``.
                - ``valid_component_indicator`` is set to ``np.array([True, True, False])``.

        Input:
            - a numpy array containing the data to be reversed.

        Ouput:
            - a numpy array with the transformed data.
        """
        # Setup
        transformer = ClusterBasedNormalizer(max_clusters=3)
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

        transformer.valid_component_indicator = np.array([True, True, False])
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
            - create an instance of the ``ClusterBasedNormalizer`` where the ``output_columns``
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
        transformer = ClusterBasedNormalizer(max_clusters=3)
        transformer.output_columns = ['col.normalized', 'col.component']
        reversed_data = np.array([0.01, 0.02, -0.01, -0.01, 0.0, 0.99, 0.97, 1.02, 1.03, 0.97])
        transformer.null_transformer = Mock()
        transformer.null_transformer.reverse_transform.return_value = reversed_data
        transformer._reverse_transform_helper = Mock()
        transformer._reverse_transform_helper.return_value = reversed_data

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

    def test__reverse_transform_missing_value_replacement_missing_value_replacement_from_col(self):
        """Test ``_reverse_transform`` with ``np.nan`` values.

        Validate that the method correctly calls ``_reverse_transform_helper`` and produces the
        appropriate output when passed a numpy array containing ``np.nan`` values.
        """
        # Setup
        transformer = ClusterBasedNormalizer(
            missing_value_generation='from_column',
            max_clusters=3
        )
        transformer.output_columns = ['col.normalized', 'col.component']
        transformer._reverse_transform_helper = Mock()
        transformer._reverse_transform_helper.return_value = np.array([
            0.68351419, 0.67292805, 0.66234274, 0.66234274, 0.67292805,
            0.63579893, 0.62239389, 0.67292805, 0.67292805, 0.62239389
        ])

        transformer.null_transformer = NullTransformer(
            'mean',
            missing_value_generation='from_column'
        )
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

    def test__reverse_transform_missing_value_replacement_missing_value_replacement_random(self):
        """Test ``_reverse_transform`` with ``np.nan`` values.

        Validate that the method correctly calls ``_reverse_transform_helper`` and produces the
        appropriate output when passed a numpy array containing ``np.nan`` values.
        """
        # Setup
        transformer = ClusterBasedNormalizer(
            missing_value_generation='from_column',
            max_clusters=3
        )
        transformer.output_columns = ['col.normalized', 'col.component']
        transformer._reverse_transform_helper = Mock()
        transformer._reverse_transform_helper.return_value = np.array([
            0.68351419, 0.67292805, 0.66234274, 0.66234274, 0.67292805,
            0.63579893, 0.62239389, 0.67292805, 0.67292805, 0.62239389
        ])

        transformer.null_transformer = NullTransformer('mean', missing_value_generation='random')
        transformer.null_transformer.fit(pd.Series([0, np.nan]))
        transformer.null_transformer.reverse_transform = Mock()
        transformer.null_transformer.reverse_transform.return_value = np.array(
            [np.nan, np.nan, np.nan, np.nan, np.nan, 0.635799, np.nan, 0.672928, 0.672928, np.nan]
        )

        data = np.array([
            [-0.033, -0.046, -0.058, -0.058, -0.046, 0.134, 0.122, -0.046, -0.046, 0.122],
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 1],
            [1, 1, 1, 1, 1, 0, 0, 0, 1, 0]
        ]).transpose()

        # Run
        output = transformer._reverse_transform(data)

        # Asserts
        expected = pd.Series(
            [np.nan, np.nan, np.nan, np.nan, np.nan, 0.63, np.nan, 0.67, 0.67, np.nan]
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
