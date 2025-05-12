"""Tools to generate strings from regular expressions."""

import datetime
import logging
import re
import string
import sys
import warnings
from collections import defaultdict
from decimal import Decimal

import numpy as np
import pandas as pd
from dateutil import parser
from dateutil.parser import ParserError, UnknownTimezoneWarning
from dateutil.tz import UTC

import sre_parse  # isort:skip

LOGGER = logging.getLogger(__name__)

MAX_DECIMALS = sys.float_info.dig
DEPRECATED_SDTYPES_MAPPING = {'text': 'id'}


def _literal(character, max_repeat):
    del max_repeat
    return iter([chr(character)]), 1


def _in(options, max_repeat):
    generators = []
    sizes = []
    for option, args in options:
        generator, size = _GENERATORS[option](args, max_repeat)
        generators.append(generator)
        sizes.append(size)

    return (value for generator in generators for value in generator), np.sum(sizes)


def _range(options, max_repeat):
    del max_repeat
    min_value, max_value = options
    max_value += 1
    return (chr(value) for value in range(min_value, max_value)), max_value - min_value


def _any(options, max_repeat):
    del options
    del max_repeat
    return iter(string.printable), len(string.printable)


def _max_repeat(options, max_repeat):
    min_, max_, options = options
    if max_ == sre_parse.MAXREPEAT:
        max_ = max_repeat

    option, args = options[0]
    _, size = _GENERATORS[option](args, max_repeat)

    generators = []
    sizes = []
    for repeat in range(min_, max_ + 1):
        if repeat:
            sizes.append(pow(int(size), repeat, 2**63 - 1))
            repeat_generators = [
                (_GENERATORS[option](args, max_repeat)[0], option, args) for _ in range(repeat)
            ]
            generators.append(_from_generators(repeat_generators, max_repeat))

    return (value for generator in generators for value in generator), np.sum(sizes) + int(
        min_ == 0
    )


def _category_chars(regex):
    return [char for char in string.printable if regex.match(char)]


_CATEGORIES = {
    sre_parse.CATEGORY_SPACE: _category_chars(re.compile(r'\s')),
    sre_parse.CATEGORY_NOT_SPACE: _category_chars(re.compile(r'\S')),
    sre_parse.CATEGORY_DIGIT: _category_chars(re.compile(r'\d')),
    sre_parse.CATEGORY_NOT_DIGIT: _category_chars(re.compile(r'\D')),
    sre_parse.CATEGORY_WORD: _category_chars(re.compile(r'\w')),
    sre_parse.CATEGORY_NOT_WORD: _category_chars(re.compile(r'\W')),
}


def _category(category, max_repeat):
    del max_repeat
    characters = _CATEGORIES[category]
    return iter(characters), len(characters)


_GENERATORS = {
    sre_parse.LITERAL: _literal,
    sre_parse.IN: _in,
    sre_parse.RANGE: _range,
    sre_parse.ANY: _any,
    sre_parse.MAX_REPEAT: _max_repeat,
    sre_parse.CATEGORY: _category,
}


def _from_generators(generators, max_repeat):
    previous = [None] + [next(generator) for generator, _, _ in generators[1:]]

    remaining = True
    while remaining:
        generated = []
        for index, (generator, option, args) in enumerate(generators):
            remaining = True
            try:
                value = next(generator)
                generated.append(value)
                previous[index] = value
                generated.extend(previous[index + 1 :])
                break
            except StopIteration:
                generator = _GENERATORS[option](args, max_repeat)[0]
                generators[index] = generator, option, args
                value = next(generator)
                previous[index] = value
                generated.append(value)
                remaining = False

        if remaining:
            yield ''.join(reversed(generated))


def _cast_to_type(data, dtype):
    if isinstance(data, pd.Series):
        data = data.apply(dtype)
    elif isinstance(data, (np.ndarray, list)):
        data = np.array([dtype(value) for value in data])
    else:
        data = dtype(data)

    return data


def strings_from_regex(regex, max_repeat=16):
    """Generate strings that match the given regular expression.

    The output is a generator that produces regular expressions that match
    the indicated regular expressions alongside an integer indicating the
    total length of the generator.

    WARNING: Subpatterns are currently not supported.

    Args:
        regex (str):
            String representing a valid python regular expression.
        max_repeat (int):
            Maximum number of repetitions to produce when the regular
            expression allows an infinte amount. Defaults to 16.

    Returns:
        tuple:
            * Generator that produces strings that match the given regex.
            * Total length of the generator.
    """
    parsed = sre_parse.parse(regex, flags=sre_parse.SRE_FLAG_UNICODE)
    generators = []
    sizes = []
    for option, args in reversed(parsed):
        if option != sre_parse.AT:
            generator, size = _GENERATORS[option](args, max_repeat)
            generators.append((generator, option, args))
            sizes.append(size)

    return _from_generators(generators, max_repeat), np.prod(sizes, dtype=np.complex128).real


def fill_nan_with_none(data):
    """Replace all nan values with None.

    Args:
        data (pd.DataFrame or pd.Series)

    Returns:
        data:
            Original data with nan values replaced by None.
    """
    return data.infer_objects().fillna(np.nan).replace([np.nan], [None])


def flatten_column_list(column_list):
    """Flatten a list of columns.

    Args:
        column_list (list):
            List of columns to flatten.

    Returns:
        list:
            Flattened list of columns.
    """
    flattened = []
    for column in column_list:
        if isinstance(column, tuple):
            flattened.extend(column)
        else:
            flattened.append(column)

    return flattened


def check_nan_in_transform(data, dtype):
    """Check if there are null values in the transformed data.

    Args:
        data (pd.Series or numpy.ndarray):
            Data that has been transformed.
        dtype (str):
            Data type of the transformed data.
    """
    if pd.isna(data).any().any():
        message = (
            'There are null values in the transformed data. The reversed '
            'transformed data will contain null values'
        )
        is_integer = pd.api.types.is_integer_dtype(dtype)
        if is_integer:
            message += " of type 'float'."
        else:
            message += '.'

        warnings.warn(message)


def try_convert_to_dtype(data, dtype):
    """Try to convert data to a given dtype.

    Args:
        data (pd.Series or numpy.ndarray):
            Data to convert.
        dtype (str):
            Data type to convert to.

    Returns:
        data:
            Data converted to the given dtype.
    """
    try:
        data = data.astype(dtype)
    except ValueError as error:
        is_integer = pd.api.types.is_integer_dtype(dtype)
        if is_integer:
            data = data.astype(float)
        else:
            raise error

    return data


def learn_rounding_digits(data):
    """Learn the number of digits to round data to.

    Args:
        data (pd.Series):
            Data to learn the number of digits to round to.

    Returns:
        int or None:
            Number of digits to round to.
    """
    # check if data has any decimals
    name = data.name
    if str(data.dtype).endswith('[pyarrow]'):
        data = data.to_numpy()
    roundable_data = data[~(np.isinf(data.astype(float)) | pd.isna(data))]

    # Empty dataset
    if len(roundable_data) == 0:
        return None

    if roundable_data.dtype == 'object':
        roundable_data = roundable_data.astype(float)

    # Try to round to fewer digits
    highest_int = int(np.max(np.abs(roundable_data)))
    most_digits = len(str(highest_int)) if highest_int != 0 else 0
    max_decimals = max(0, MAX_DECIMALS - most_digits)
    if (roundable_data == roundable_data.round(max_decimals)).all():
        for decimal in range(max_decimals + 1):
            if (roundable_data == roundable_data.round(decimal)).all():
                return decimal

    # Can't round, not equal after MAX_DECIMALS digits of precision
    LOGGER.info(
        "No rounding scheme detected for column '%s'. Data will not be rounded.",
        name,
    )
    return None


def logit(data, low, high):
    """Apply a logit function to the data using ``low`` and ``high``.

    Args:
        data (pd.Series, pd.DataFrame, np.array, int, or float):
            Data to apply the logit function to.
        low (pd.Series, np.array, int, or float):
            Low value/s to use when scaling.
        high (pd.Series, np.array, int, or float):
            High value/s to use when scaling.

    Returns:
        Logit scaled version of the input data.
    """
    data = (data - low) / (high - low)
    data = _cast_to_type(data, Decimal)
    data = data * Decimal(0.95) + Decimal(0.025)
    data = _cast_to_type(data, float)
    return np.log(data / (1.0 - data))


def sigmoid(data, low, high):
    """Apply a sigmoid function to the data using ``low`` and ``high``.

    Args:
        data (pd.Series, pd.DataFrame, np.array, int, float or datetime):
            Data to apply the logit function to.
        low (pd.Series, np.array, int, float or datetime):
            Low value/s to use when scaling.
        high (pd.Series, np.array, int, float or datetime):
            High value/s to use when scaling.

    Returns:
        Sigmoid transform of the input data.
    """
    data = 1 / (1 + np.exp(-data))
    data = _cast_to_type(data, Decimal)
    data = (data - Decimal(0.025)) / Decimal(0.95)
    data = _cast_to_type(data, float)
    data = data * (high - low) + low

    return data


class WarnDict(dict):
    """Custom dictionary to raise a deprecation warning."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._warned = defaultdict()

    def get(self, sdtype):
        """Return the value for sdtype if sdtype is in the dictionary, else default.

        If the sdtype is `text` raises a `DeprecationWarning` stating that it will be
        phased out.
        """
        if sdtype in DEPRECATED_SDTYPES_MAPPING and not self._warned.get(sdtype):
            new_sdtype = DEPRECATED_SDTYPES_MAPPING.get(sdtype)
            warnings.warn(
                f"The sdtype '{sdtype}' is deprecated and will be phased out. "
                f"Please use '{new_sdtype}' instead.",
                DeprecationWarning,
            )
            self._warned[sdtype] = True

        return super().get(sdtype)

    def __getitem__(self, sdtype):
        """Return the value for sdtype if sdtype is in the dictionary.

        If the sdtype is `text` raises a `DeprecationWarning` stating that it will be
        phased out.
        """
        return self.get(sdtype)


def _handle_enforce_uniqueness_and_cardinality_rule(enforce_uniqueness, cardinality_rule):
    if enforce_uniqueness is not None:
        warnings.warn(
            "The 'enforce_uniqueness' parameter is no longer supported. "
            "Please use the 'cardinality_rule' parameter instead.",
            FutureWarning,
        )
        if enforce_uniqueness and cardinality_rule is None:
            return 'unique'

    if cardinality_rule not in ['unique', 'match', 'scale', None]:
        raise ValueError(
            "The 'cardinality_rule' parameter must be one of 'unique', 'match', 'scale', or None."
        )

    return cardinality_rule


def _extract_timezone_from_a_string(dt_str):
    if not isinstance(dt_str, str):
        dt_str = str(dt_str)

    match = re.search(r'\b([A-Z]{2,5})\b', dt_str)
    return match.group(1) if match else None


def _safe_parse_datetime(value, warn=False, datetime_format=None):
    """Safely parse a value into a datetime object, handling invalid inputs.

    Converts the input `value` into a `datetime.datetime` object using `dateutil.parser.parse`.
    If the input is already a `pandas.Timestamp` or `datetime.datetime`, it is returned unchanged.
    Unrecognized timezones are converted to UTC, with an optional warning. Returns `pd.NaT` if
    parsing fails or the input is invalid.

    Args:
        value (Any):
            Value to parse into a datetime. Accepts strings, `pandas.Timestamp`,
            `datetime.datetime`, or types compatible with `dateutil.parser.parse`.
        warn (bool):
            If True, warns when an unrecognized timezone is converted to UTC.
            Defaults to False.
        datetime_format (str):
            Format of the datetime string.
            Defaults to None.

    Returns:
        datetime.datetime | pd.NaT:
            Parsed `datetime.datetime` object with unrecognized timezones
            set to UTC, or `pd.NaT` if parsing fails.
    """
    if isinstance(value, (pd.Timestamp, datetime.datetime)):
        return value

    try:
        with warnings.catch_warnings(record=True) as captured_warnings:
            try:
                dt = parser.parse(value)

            # Strings of large numbers cause parser.parse to overflow
            except (OverflowError, ParserError) as error:
                if not datetime_format:
                    raise error
                value = str(pd.to_datetime(value, format=datetime_format))
                dt = parser.parse(value)

        if any(issubclass(warned.category, UnknownTimezoneWarning) for warned in captured_warnings):
            input_timezone = _extract_timezone_from_a_string(value)
            if warn:
                warnings.warn(
                    f"Timezone '{input_timezone}' is not understood so it will be converted "
                    "to 'UTC'.",
                )

            return dt.replace(tzinfo=UTC)

        return dt

    except (ValueError, TypeError, AttributeError, OverflowError):
        return None


def _get_utc_offset(dt):
    try:
        return dt.utcoffset()
    except AttributeError:
        return None


def data_has_multiple_timezones(data, datetime_format=None):
    """Check if a Series of datetime values contains multiple timezones.

    Args:
        data (pd.Series):
            Series of datetime strings or datetime objects.
        datetime_format (str):
            Format of the datetime string.
            Defaults to None.

    Returns:
        bool:
            True if multiple timezones are detected, False otherwise.
    """
    try:
        parsed_datetimes = data.apply(
            _safe_parse_datetime, datetime_format=datetime_format
        ).dropna()
        offsets = parsed_datetimes.apply(_get_utc_offset).dropna()
        return offsets.nunique() > 1

    except ValueError:
        return False


def _get_cardinality_frequency(data):
    """Get number of repetitions of values in the data and their frequencies."""
    value_counts = data.value_counts(dropna=False)
    repetition_counts = value_counts.value_counts().sort_index()
    total = repetition_counts.sum()
    frequencies = (repetition_counts / total).tolist()
    repetitions = repetition_counts.index.tolist()

    return repetitions, frequencies


def _sample_repetitions(num_samples, value, data_cardinality_scale, remaining_samples):
    """Sample a number of repetitions for a given value."""
    repetitions = np.random.choice(
        data_cardinality_scale['num_repetitions'],
        p=data_cardinality_scale['frequency'],
    )
    if repetitions <= num_samples:
        samples = [value] * repetitions
    else:
        samples = [value] * num_samples
        remaining_samples['repetitions'] = repetitions - num_samples
        remaining_samples['value'] = value

    return samples, remaining_samples
