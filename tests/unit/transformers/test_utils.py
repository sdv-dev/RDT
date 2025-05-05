import datetime
import re
import sre_parse
import warnings
from decimal import Decimal
from sre_constants import MAXREPEAT
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from dateutil import parser, tz

from rdt.transformers.utils import (
    WarnDict,
    _any,
    _cast_to_type,
    _extract_timezone_from_a_string,
    _get_utc_offset,
    _handle_enforce_uniqueness_and_cardinality_rule,
    _max_repeat,
    _safe_parse_datetime,
    check_nan_in_transform,
    data_has_multiple_timezones,
    fill_nan_with_none,
    flatten_column_list,
    learn_rounding_digits,
    logit,
    sigmoid,
    strings_from_regex,
    try_convert_to_dtype,
)


def test__cast_to_type():
    """Test the ``_cast_to_type`` function.

    Given ``pd.Series``, ``np.array`` or just a numeric value, it should
    cast it to the given ``type``.

    Input:
        - pd.Series
        - np.array
        - numeric
        - Type
    Output:
        The values should be casted to the expected ``type``.
    """
    # Setup
    value = 88
    series = pd.Series([1, 2, 3])
    array = np.array([1, 2, 3])

    # Run
    res_value = _cast_to_type(value, float)
    res_series = _cast_to_type(series, float)
    res_array = _cast_to_type(array, float)

    # Assert
    assert isinstance(res_value, float)
    assert res_series.dtype == float
    assert res_array.dtype == float


def test_strings_from_regex_literal():
    generator, size = strings_from_regex('abcd', max_repeat=16)

    assert size == 1
    assert list(generator) == ['abcd']


def test_strings_from_regex_digit():
    generator, size = strings_from_regex('[0-9]')

    assert size == 10
    assert list(generator) == [
        '0',
        '1',
        '2',
        '3',
        '4',
        '5',
        '6',
        '7',
        '8',
        '9',
    ]


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
    column_list = [
        'column1',
        ('column2', 'column3'),
        'column4',
        ('column5',),
        'column6',
    ]

    # Run
    flattened_list = flatten_column_list(column_list)

    # Assert
    expected_flattened_list = [
        'column1',
        'column2',
        'column3',
        'column4',
        'column5',
        'column6',
    ]
    assert flattened_list == expected_flattened_list


@pytest.mark.filterwarnings('error')
def test_fill_nan_with_none_no_warning():
    """Test the `fill_nan_with_none`` does not generate a FutureWarning.

    Based on the issue [#793](https://github.com/sdv-dev/RDT/issues/793).
    """
    # Setup
    series = pd.Series([1.0, 2.0, 3.0, np.nan], dtype='object')

    # Run
    result = fill_nan_with_none(series)

    # Assert
    expected = pd.Series([1.0, 2.0, 3.0, None], dtype='object')
    pd.testing.assert_series_equal(result, expected)


def test_check_nan_in_transform():
    """Test ``check_nan_in_transform`` method.

    If there nan in the data, a warning should be raised.
    If the data was integer, it should be converted to float.
    """
    # Setup
    transformed = pd.Series([
        0.1026,
        0.1651,
        np.nan,
        0.3116,
        0.6546,
        0.8541,
        0.7041,
    ])
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


@pytest.mark.parametrize(
    'test_data, expected_digits',
    [
        # Basic decimal places test
        (pd.Series([10, 0.0, 0.1, 0.12, 0.123, np.nan]), 3),
        # Large numbers with decimals
        (pd.Series([1234567890123456.7, 12345678901234567.89, 123456789012345678.901]), None),
        # Consistent single decimal place
        (
            pd.Series([
                1.1,
                11.1,
                111.1,
                1111.1,
                11111.1,
                111111.1,
                1111111.1,
                11111111.1,
                111111111.1,
                1111111111.1,
                11111111111.1,
                111111111111.1,
                1111111111111.1,
                11111111111111.1,
            ]),
            1,
        ),
        # Various precision tests
        (pd.Series([123456789012345.6789]), None),
        (pd.Series([12345678901.234]), 3),
        (pd.Series([12345678901.2345]), 4),
        (pd.Series([0.1234567890123456]), None),
        (pd.Series([0.123456789012345]), 15),
        # Integer tests
        (pd.Series([123456789012345]), 0),
        (pd.Series([12345678901234567890]), 0),
        # Mixed number and edge cases
        (pd.Series([12345678901234567890, 1.123]), None),
        (pd.Series([0.000000000000000000001]), None),
        (pd.Series([-12345678901234, -1.123]), None),
        (pd.Series([-0.123456789012345]), 15),
    ],
)
def test_learn_rounding_digits(test_data, expected_digits):
    """Test learn_rounding_digits for various test cases."""
    # Run
    result = learn_rounding_digits(test_data)

    # Assert
    assert result == expected_digits


def test_learn_rounding_digits_object_dtype():
    """Test learn_rounding_digits for object dtype."""
    # Setup
    data = pd.Series(['1.1', '1.11', np.nan], dtype='object')

    # Run
    result = learn_rounding_digits(data)

    # Assert
    assert result == 2


def test_learn_rounding_digits_code_coverage():
    """Test learn_rounding_digits for code coverage."""
    # Setup
    data = pd.Series([np.inf, -np.inf, np.nan])

    # Run
    result = learn_rounding_digits(data)

    # Assert
    assert result is None


def test_learn_rounding_digits_pyarrow():
    """Test it works with pyarrow."""
    # Setup
    try:
        data = pd.Series(range(10), dtype='int64[pyarrow]')
    except TypeError:
        pytest.skip("Skipping as old numpy/pandas versions don't support arrow")

    # Run
    output = learn_rounding_digits(data)

    # Assert
    assert output == 0


def test_learn_rounding_digits_pyarrow_float():
    """Test it learns the proper amount of digits with pyarrow."""
    # Setup
    try:
        data = pd.Series([0.5, 0.19, 3], dtype='float64[pyarrow]')
    except TypeError:
        pytest.skip("Skipping as old numpy/pandas versions don't support arrow")

    # Run
    output = learn_rounding_digits(data)

    # Assert
    assert output == 2


def test_learn_rounding_digits_nullable_numerical_pandas_dtypes():
    """Test that ``learn_rounding_digits`` supports the nullable numerical pandas dtypes."""
    # Setup
    data = pd.DataFrame({
        'Int8': pd.Series([1, 2, -3, pd.NA, None, pd.NA], dtype='Int8'),
        'Int16': pd.Series([1, 2, -3, pd.NA, None, pd.NA], dtype='Int16'),
        'Int32': pd.Series([1, 2, -3, pd.NA, None, pd.NA], dtype='Int32'),
        'Int64': pd.Series([1, 2, -3, pd.NA, None, pd.NA], dtype='Int64'),
        'Float32': pd.Series([1.12, 2.23, 3.33, pd.NA, None, pd.NA], dtype='Float32'),
        'Float64': pd.Series([1.1234, 2.2345, 3.323, pd.NA, None, pd.NA], dtype='Float64'),
    })
    expected_output = {
        'Int8': 0,
        'Int16': 0,
        'Int32': 0,
        'Int64': 0,
        'Float32': 2,
        'Float64': 4,
    }

    # Run and Assert
    for column in data.columns:
        output = learn_rounding_digits(data[column])
        assert output == expected_output[column]


def test_learn_rounding_digits_pyarrow_to_numpy():
    """Test that ``learn_rounding_digits`` works with pyarrow to numpy conversion."""
    # Setup
    data = Mock()
    data.dtype = 'int64[pyarrow]'
    data.to_numpy.return_value = np.array([1, 2, 3])

    # Run
    learn_rounding_digits(data)

    # Assert
    assert data.to_numpy.called


def test_logit():
    """Test the ``logit`` function.

    Setup:
        - Compute ``expected_res`` with the ``high`` and ``low`` values.
    Input:
        - ``data`` a number.
        - ``low`` and ``high`` numbers.
    Output:
        The result of the scaled logit.
    """
    # Setup
    high, low = 100, 49
    _data = (88 - low) / (high - low)
    _data = Decimal(_data) * Decimal(0.95) + Decimal(0.025)
    _data = float(_data)
    expected_res = np.log(_data / (1.0 - _data))

    data = 88

    # Run
    res = logit(data, low, high)

    # Assert

    assert res == expected_res


def test_sigmoid():
    """Test the ``sigmoid`` function.

    Setup:
        - Compute ``expected_res`` with the ``high`` and ``low`` values.
    Input:
        - ``data`` a number.
        - ``low`` and ``high`` numbers.
    Output:
        The result of sigmoid.
    """
    # Setup
    high, low = 100, 49
    _data = data = 1.1064708752806303

    _data = 1 / (1 + np.exp(-data))
    _data = (Decimal(_data) - Decimal(0.025)) / Decimal(0.95)
    _data = float(_data)
    expected_res = _data * (high - low) + low

    # Run
    res = sigmoid(data, low, high)

    # Assert
    assert res == expected_res


def test_warn_dict():
    """Test that ``WarnDict`` will raise a warning when called with `text`."""
    # Setup
    instance = WarnDict()
    instance['text'] = 'text_transformer'

    # Run
    warning_msg = "The sdtype 'text' is deprecated and will be phased out. Please use 'id' instead."
    with pytest.warns(DeprecationWarning, match=warning_msg):
        result_access = instance['text']

    # Run second time and no warning gets shown
    with warnings.catch_warnings(record=True) as record:
        result_access_no_warn = instance['text']
        result_get_no_warn = instance.get('text')

    # Assert
    assert len(record) == 0
    assert result_access == 'text_transformer'
    assert result_access_no_warn == 'text_transformer'
    assert result_get_no_warn == 'text_transformer'


def test_warn_dict_get():
    """Test that ``WarnDict`` will raise a warning when called with `text`."""
    # Setup
    instance = WarnDict()
    instance['text'] = 'text_transformer'

    # Run
    warning_msg = "The sdtype 'text' is deprecated and will be phased out. Please use 'id' instead."
    with pytest.warns(DeprecationWarning, match=warning_msg):
        result_access = instance.get('text')

    # Run second time and no warning gets shown
    with warnings.catch_warnings(record=True) as record:
        result_access_no_warn = instance['text']
        result_get_no_warn = instance.get('text')

    # Assert
    assert len(record) == 0
    assert result_access == 'text_transformer'
    assert result_access_no_warn == 'text_transformer'
    assert result_get_no_warn == 'text_transformer'


def test__handle_enforce_uniqueness_and_cardinality_rule():
    """Test that ``_handle_enforce_uniqueness_and_cardinality_rule`` works as expected."""
    # Setup
    enforce_uniqueness = None
    cardinality_rule = None
    expected_message = re.escape(
        "The 'enforce_uniqueness' parameter is no longer supported. "
        "Please use the 'cardinality_rule' parameter instead."
    )

    # Run
    result_1 = _handle_enforce_uniqueness_and_cardinality_rule(enforce_uniqueness, cardinality_rule)
    with pytest.warns(FutureWarning, match=expected_message):
        result_2 = _handle_enforce_uniqueness_and_cardinality_rule(True, None)

    with pytest.warns(FutureWarning, match=expected_message):
        result_3 = _handle_enforce_uniqueness_and_cardinality_rule(True, 'other')

    # Assert
    assert result_1 is None
    assert result_2 == 'unique'
    assert result_3 == 'other'


def test__extract_timezone_from_a_string_with_valid_timezone():
    """Test that `_extract_timezone_from_a_string` extracts a valid timezone from a string."""
    # Setup
    dt_str = '2023-10-15 14:30:00 XYZ'

    # Run
    result = _extract_timezone_from_a_string(dt_str)

    # Assert
    assert result == 'XYZ'


def test__extract_timezone_from_a_string_with_no_timezone():
    """Test that `_extract_timezone_from_a_string` returns None when no timezone is present."""
    # Setup
    dt_str = '2023-10-15 14:30:00'

    # Run
    result = _extract_timezone_from_a_string(dt_str)

    # Assert
    assert result is None


def test__extract_timezone_from_a_string_with_timezone_object():
    """Test that `_extract_timezone_from_a_string` returns None when no timezone is present."""
    # Setup
    timestamp = pd.to_datetime('2021-04-04 23:00:10 UTC')

    # Run
    result = _extract_timezone_from_a_string(timestamp.tz)

    # Assert
    assert result == 'UTC'


def test__safe_parse_datetime():
    """Test the ``_safe_parse_datetime`` function.

    Setup:
        - Use a string datetime with timezone offset.
        - Use a pandas.Timestamp and a datetime.datetime object.
    Input:
        - A mix of string and datetime-like inputs.
    Output:
        - Parsed datetime or original datetime-like value.
    """
    # Setup
    str_input = '2023-01-01 12:00:00+0200'
    dt_input = datetime.datetime(2023, 1, 1, 12, 0)
    ts_input = pd.Timestamp(dt_input)

    # Run
    res_str = _safe_parse_datetime(str_input)
    res_dt = _safe_parse_datetime(dt_input)
    res_ts = _safe_parse_datetime(ts_input)
    res_invalid = _safe_parse_datetime('not-a-date')

    # Assert
    assert res_str.isoformat() == '2023-01-01T12:00:00+02:00'
    assert res_dt == dt_input
    assert res_ts == ts_input
    assert res_invalid is None


def test__safe_parse_datetime_with_unrecognized_timezone_and_warning():
    """Test that `_safe_parse_datetime` handles unrecognized timezone."""
    # Setup
    value = '2023-10-15 14:30:00 XYZ'
    warn = True
    expected_dt = parser.parse('2023-10-15 14:30:00').replace(tzinfo=tz.tzoffset('UTC', 0))

    # Run
    warning_msg = "Timezone 'XYZ' is not understood so it will be converted to 'UTC'."
    with pytest.warns(UserWarning, match=warning_msg):
        result = _safe_parse_datetime(value, warn=warn)

    # Assert
    assert result == expected_dt


def test__get_utc_offset():
    """Test the ``_get_utc_offset`` function.

    Setup:
        - Use aware and naive datetime objects.
    Input:
        - Datetime objects with and without timezone info.
    Output:
        - UTC offset as timedelta or None.
    """
    # Setup
    aware = datetime.datetime(
        2023, 1, 1, 12, 0, tzinfo=datetime.timezone(datetime.timedelta(hours=2))
    )
    naive = datetime.datetime(2023, 1, 1, 12, 0)

    # Run
    res_aware = _get_utc_offset(aware)
    res_naive = _get_utc_offset(naive)
    res_invalid = _get_utc_offset(None)

    # Assert
    assert res_aware == datetime.timedelta(hours=2)
    assert res_naive is None
    assert res_invalid is None


def test_data_has_multiple_timezones():
    """Test the ``data_has_multiple_timezones`` function.

    Setup:
        - Provide datetime strings with different timezone offsets.
    Input:
        - A pandas Series of datetime strings.
    Output:
        - True if timezones differ, False otherwise.
    """
    # Setup
    data_mixed = pd.Series(['2023-01-01 12:00:00+0200', '2023-01-01 12:00:00+0300'])
    data_uniform = pd.Series(['2023-01-01 12:00:00+0200', '2023-01-02 12:00:00+0200'])
    data_invalid = pd.Series(['invalid', 'still bad'])

    # Run
    res_mixed = data_has_multiple_timezones(data_mixed)
    res_uniform = data_has_multiple_timezones(data_uniform)
    res_invalid = data_has_multiple_timezones(data_invalid)

    # Assert
    assert res_mixed is True
    assert res_uniform is False
    assert res_invalid is False


@patch('rdt.transformers.utils._safe_parse_datetime')
def test_data_has_multiple_timezones_error_out(mock__safe_parse):
    """Test the ``data_has_multiple_timezones`` function.

    Setup:
        - Provide datetime strings with different timezone offsets.
    Input:
        - A pandas Series of datetime strings.
    Output:
        - True if timezones differ, False otherwise.
    """
    # Setup
    mock__safe_parse.side_effect = ValueError('Bad input')
    data_invalid = pd.Series(['invalid', 'still bad'])

    # Run
    res_invalid = data_has_multiple_timezones(data_invalid)

    # Assert
    assert res_invalid is False
