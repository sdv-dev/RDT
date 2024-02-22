import sre_parse
from sre_constants import MAXREPEAT
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from rdt.transformers.utils import (
    _any, _max_repeat, check_nan_in_transform, flatten_column_list, learn_rounding_digits,
    strings_from_regex, try_convert_to_dtype)


def test_strings_from_regex_literal():
    generator, size = strings_from_regex('abcd', max_repeat=16)

    assert size == 1
    assert list(generator) == ['abcd']


def test_strings_from_regex_digit():
    generator, size = strings_from_regex('[0-9]')

    assert size == 10
    assert list(generator) == ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def test_strings_from_regex_repeat_literal():
    generator, size = strings_from_regex('a{1,3}')

    assert size == 3
    assert list(generator) == ['a', 'aa', 'aaa']


def test_strings_from_regex_repeat_digit():
    generator, size = strings_from_regex(r'\d{1,3}')

    assert size == 1110

    strings = list(generator)
    assert strings[0] == '0'
    assert strings[-1] == '999'


def test__any():
    options = {'a': 1}
    max_repeat = 5
    _any(options, max_repeat)


def test___max_repeat():
    options = (0, MAXREPEAT, [(sre_parse.LITERAL, 10)])
    _max_repeat(options, 16)


def test_strings_from_regex_very_large_regex():
    """Ensure that ``size`` of a very large regex is still computable."""
    very_large_regex = '[0-9a-zA-Z]{9}-[0-9a-zA-Z]{4}-[0-9a-zA-Z]{9}-[0-9a-zA-Z]{9}-[0-9a-z]{12}'
    generator, size = strings_from_regex(very_large_regex, max_repeat=16)

    assert size == 173689027553046619421110743915454114823342474255318764491341273608665169920
    [next(generator) for _ in range(100_000)]


def test_flatten_column_list():
    """Test `flatten_column_list` function."""
    # Setup
    column_list = ['column1', ('column2', 'column3'), 'column4', ('column5',), 'column6']

    # Run
    flattened_list = flatten_column_list(column_list)

    # Assert
    expected_flattened_list = ['column1', 'column2', 'column3', 'column4', 'column5', 'column6']
    assert flattened_list == expected_flattened_list


def test_check_nan_in_transform():
    """Test ``check_nan_in_transform`` method.

    If there nan in the data, a warning should be raised.
    If the data was integer, it should be converted to float.
    """
    # Setup
    transformed = pd.Series([0.1026, 0.1651, np.nan, 0.3116, 0.6546, 0.8541, 0.7041])
    data_without_nans = pd.DataFrame({
        'col 1': [1, 2, 3],
        'col 2': [4, 5, 6],
    })

    # Run and Assert
    check_nan_in_transform(data_without_nans, 'float')
    expected_message = (
        'There are null values in the transformed data. The reversed '
        'transformed data will contain null values'
    )
    expected_message_object = expected_message + '.'
    expected_message_integer = expected_message + " of type 'float'."
    with pytest.warns(UserWarning, match=expected_message_object):
        check_nan_in_transform(transformed, 'object')

    with pytest.warns(UserWarning, match=expected_message_integer):
        check_nan_in_transform(transformed, 'int')


def test_try_convert_to_dtype():
    """Test ``try_convert_to_dtype`` method.

    If the data can be converted to the specified dtype, it should be converted.
    If the data cannot be converted, a ValueError should be raised.
    Should allow to convert integer with NaNs to float.
    """
    # Setup
    data_int_with_nan = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
    data_not_convertible = pd.Series(['a', 'b', 'c', 'd', 'e'])

    # Run
    output_convertibe = try_convert_to_dtype(data_int_with_nan, 'str')
    output_int_with_nan = try_convert_to_dtype(data_int_with_nan, 'int')
    with pytest.raises(ValueError, match="could not convert string to float: 'a'"):
        try_convert_to_dtype(data_not_convertible, 'int')

    with pytest.raises(ValueError, match="could not convert string to float: 'a'"):
        try_convert_to_dtype(data_not_convertible, 'float')

    # Assert
    expected_data_with_nan = pd.Series([1, 2, np.nan, 4, 5])
    expected_data_convertibe = pd.Series(['1.0', '2.0', 'nan', '4.0', '5.0'])
    pd.testing.assert_series_equal(output_int_with_nan, expected_data_with_nan)
    pd.testing.assert_series_equal(output_convertibe, expected_data_convertibe)


@patch('rdt.transformers.utils.LOGGER')
def test_learn_rounding_digits_more_than_15_decimals(logger_mock):
    """Test the learn_rounding_digits method with more than 15 decimals.

    If the data has more than 15 decimals, return None and raise warning.
    """
    # Setup
    data = pd.Series(np.random.random(size=10).round(20), name='col')

    # Run
    output = learn_rounding_digits(data)

    # Assert
    logger_msg = "No rounding scheme detected for column '%s'. Data will not be rounded."
    logger_mock.info.assert_called_once_with(logger_msg, 'col')
    assert output is None


def test_learn_rounding_digits_less_than_15_decimals():
    """Test the learn_rounding_digits method with less than 15 decimals.

    If the data has less than 15 decimals, the maximum number of decimals
    should be returned.

    Input:
    - An array that contains floats with a maximum of 3 decimals and a
        NaN.
    Output:
    - 3
    """
    data = pd.Series(np.array([10, 0., 0.1, 0.12, 0.123, np.nan]))

    output = learn_rounding_digits(data)

    assert output == 3


def test_learn_rounding_digits_negative_decimals_float():
    """Test the learn_rounding_digits method with floats multiples of powers of 10.

    If the data has all multiples of 10 the output should be 0.

    Input:
    - An array that contains floats that are multiples of powers of 10, 100 and 1000 and a NaN.
    """
    data = pd.Series(np.array([1230., 12300., 123000., np.nan]))

    output = learn_rounding_digits(data)

    assert output == 0


def test_learn_rounding_digits_negative_decimals_integer():
    """Test the learn_rounding_digits method with integers multiples of powers of 10.

    If the data has all multiples of 10 the output should be 0.

    Input:
    - An array that contains integers that are multiples of powers of 10, 100 and 1000
        and a NaN.
    """
    data = pd.Series(np.array([1230, 12300, 123000, np.nan]))

    output = learn_rounding_digits(data)

    assert output == 0


def test_learn_rounding_digits_all_missing_value_replacements():
    """Test the learn_rounding_digits method with data that is all NaNs.

    If the data is all NaNs, expect that the output is None.

    Input:
    - An array of NaN.
    Output:
    - None
    """
    data = pd.Series(np.array([np.nan, np.nan, np.nan, np.nan]))

    output = learn_rounding_digits(data)

    assert output is None
