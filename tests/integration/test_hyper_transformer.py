"""Integration tests for the HyperTransformer."""

import re
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from rdt import HyperTransformer
from rdt.errors import NotFittedError
from rdt.transformers import (
    DEFAULT_TRANSFORMERS, BaseTransformer, BinaryEncoder, FloatFormatter, FrequencyEncoder,
    OneHotEncoder, UnixTimestampEncoder, get_default_transformer, get_default_transformers)


class DummyTransformerNumerical(BaseTransformer):

    INPUT_SDTYPE = 'categorical'
    OUTPUT_SDTYPES = {
        'value': 'float'
    }

    def _fit(self, data):
        pass

    def _transform(self, data):
        return data.astype(float)

    def _reverse_transform(self, data):
        return data.astype(str)


class DummyTransformerNotMLReady(BaseTransformer):

    INPUT_SDTYPE = 'datetime'
    OUTPUT_SDTYPES = {
        'value': 'categorical',
    }

    def _fit(self, data):
        pass

    def _transform(self, data):
        # Stringify input data
        return data.astype(str)

    def _reverse_transform(self, data):
        return data.astype('datetime64')


TEST_DATA_INDEX = [4, 6, 3, 8, 'a', 1.0, 2.0, 3.0]


def get_input_data():
    datetimes = pd.to_datetime([
        np.nan,
        '2010-02-01',
        '2010-01-01',
        '2010-01-01',
        '2010-01-01',
        '2010-02-01',
        '2010-01-01',
        '2010-01-01',
    ])
    data = pd.DataFrame({
        'integer': [1, 2, 1, 3, 1, 4, 2, 3],
        'float': [0.1, 0.2, 0.1, np.nan, 0.1, 0.4, np.nan, 0.3],
        'categorical': ['a', 'a', np.nan, 'b', 'a', 'b', 'a', 'a'],
        'bool': [False, np.nan, False, True, False, True, True, False],
        'datetime': datetimes,
        'names': ['Jon', 'Arya', 'Arya', 'Jon', 'Jon', 'Sansa', 'Jon', 'Jon'],
    }, index=TEST_DATA_INDEX)

    return data


def get_transformed_data():
    datetimes = [
        1.263069e+18,
        1.264982e+18,
        1.262304e+18,
        1.262304e+18,
        1.262304e+18,
        1.264982e+18,
        1.262304e+18,
        1.262304e+18
    ]
    return pd.DataFrame({
        'integer.value': [1, 2, 1, 3, 1, 4, 2, 3],
        'float.value': [0.1, 0.2, 0.1, 0.2, 0.1, 0.4, 0.2, 0.3],
        'categorical.value': [0.3125, 0.3125, 0.9375, 0.75, 0.3125, 0.75, 0.3125, 0.3125],
        'bool.value': [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        'datetime.value': datetimes,
        'names.value': [0.3125, 0.75, 0.75, 0.3125, 0.3125, 0.9375, 0.3125, 0.3125]
    }, index=TEST_DATA_INDEX)


def get_reversed_data():
    datetimes = pd.to_datetime([
        np.nan,
        '2010-02-01',
        '2010-01-01',
        '2010-01-01',
        '2010-01-01',
        '2010-02-01',
        '2010-01-01',
        '2010-01-01',
    ])
    data = pd.DataFrame({
        'integer': [1, 2, 1, 3, 1, 4, 2, 3],
        'float': [0.1, 0.2, 0.1, np.nan, 0.1, 0.4, np.nan, 0.3],
        'categorical': ['a', 'a', np.nan, 'b', 'a', 'b', 'a', 'a'],
        'bool': [np.nan, np.nan, np.nan, True, np.nan, True, True, np.nan],
        'datetime': datetimes,
        'names': ['Jon', 'Arya', 'Arya', 'Jon', 'Jon', 'Sansa', 'Jon', 'Jon'],
    }, index=TEST_DATA_INDEX)

    return data


DETERMINISTIC_DEFAULT_TRANSFORMERS = deepcopy(DEFAULT_TRANSFORMERS)
DETERMINISTIC_DEFAULT_TRANSFORMERS['categorical'] = FrequencyEncoder


@patch('rdt.transformers.DEFAULT_TRANSFORMERS', DETERMINISTIC_DEFAULT_TRANSFORMERS)
def test_hypertransformer_default_inputs():
    """Test the HyperTransformer with default parameters.

    This tests that if default parameters are provided to the HyperTransformer,
    the ``default_transformers`` method will be used to determine which
    transformers to use for each field.

    Setup:
        - Patch the ``DEFAULT_TRANSFORMERS`` to use the ``FrequencyEncoder``
        for categorical sdtypes, so that the output is predictable.

    Input:
        - A dataframe with every sdtype.

    Expected behavior:
        - The transformed data should contain all the ML ready data.
        - The reverse transformed data should be the same as the input.
    """
    # Setup
    data = get_input_data()

    # Run
    ht = HyperTransformer()
    ht.detect_initial_config(data)
    ht.fit(data)
    transformed = ht.transform(data)
    reverse_transformed = ht.reverse_transform(transformed)

    # Assert
    expected_transformed = get_transformed_data()
    pd.testing.assert_frame_equal(transformed, expected_transformed)

    expected_reversed = get_reversed_data()
    pd.testing.assert_frame_equal(expected_reversed, reverse_transformed)

    assert isinstance(ht._transformers_tree['integer']['transformer'], FloatFormatter)
    assert ht._transformers_tree['integer']['outputs'] == ['integer.value']
    assert isinstance(ht._transformers_tree['float']['transformer'], FloatFormatter)
    assert ht._transformers_tree['float']['outputs'] == ['float.value']
    assert isinstance(ht._transformers_tree['categorical']['transformer'], FrequencyEncoder)
    assert ht._transformers_tree['categorical']['outputs'] == ['categorical.value']
    assert isinstance(ht._transformers_tree['bool']['transformer'], BinaryEncoder)
    assert ht._transformers_tree['bool']['outputs'] == ['bool.value']
    assert isinstance(ht._transformers_tree['datetime']['transformer'], UnixTimestampEncoder)
    assert ht._transformers_tree['datetime']['outputs'] == ['datetime.value']
    assert isinstance(ht._transformers_tree['names']['transformer'], FrequencyEncoder)
    assert ht._transformers_tree['names']['outputs'] == ['names.value']

    get_default_transformers.cache_clear()
    get_default_transformer.cache_clear()


def test_hypertransformer_field_transformers():
    """Test the HyperTransformer with ``field_transformers`` provided.

    This tests that the transformers specified in the ``field_transformers``
    argument are used. Any output of a transformer that is not ML ready (not
    in the ``_transform_output_sdtypes`` list) should be recursively transformed
    till it is.

    Setup:
        - The datetime column is set to use a dummy transformer that stringifies
        the input. That output is then set to use the categorical transformer.

    Input:
        - A dict mapping each field to a transformer.
        - A dataframe with every sdtype.

    Expected behavior:
        - The transformed data should contain all the ML ready data.
        - The reverse transformed data should be the same as the input.
    """
    # Setup
    config = {
        'sdtypes': {},
        'transformers': {
            'integer': FloatFormatter(missing_value_replacement='mean'),
            'float': FloatFormatter(missing_value_replacement='mean'),
            'categorical': FrequencyEncoder,
            'bool': BinaryEncoder(missing_value_replacement='mode'),
            'datetime': DummyTransformerNotMLReady,
            'datetime.value': FrequencyEncoder,
            'names': FrequencyEncoder
        }
    }

    data = get_input_data()

    # Run
    ht = HyperTransformer()
    ht.detect_initial_config(data)
    ht.set_config(config)
    ht.fit(data)
    transformed = ht.transform(data)
    reverse_transformed = ht.reverse_transform(transformed)

    # Assert
    expected_transformed = get_transformed_data()
    rename = {'datetime.value': 'datetime.value.value'}
    expected_transformed = expected_transformed.rename(columns=rename)
    transformed_datetimes = [0.9375, 0.75, 0.3125, 0.3125, 0.3125, 0.75, 0.3125, 0.3125]
    expected_transformed['datetime.value.value'] = transformed_datetimes
    pd.testing.assert_frame_equal(transformed, expected_transformed)

    expected_reversed = get_reversed_data()
    pd.testing.assert_frame_equal(expected_reversed, reverse_transformed)


def test_single_category():
    """Test that categorical variables with a single value are supported."""
    # Setup
    ht = HyperTransformer()
    data = pd.DataFrame({
        'a': ['a', 'a', 'a']
    })

    # Run
    ht.detect_initial_config(data)
    ht.update_transformers(column_name_to_transformer={
        'a': OneHotEncoder()
    })
    ht.fit(data)
    transformed = ht.transform(data)
    reverse = ht.reverse_transform(transformed)

    # Assert
    pd.testing.assert_frame_equal(data, reverse)


def test_dtype_category():
    """Test that categorical variables of dtype category are supported."""
    # Setup
    data = pd.DataFrame({'a': ['a', 'b', 'c']}, dtype='category')

    # Run
    ht = HyperTransformer()
    ht.detect_initial_config(data)
    ht.fit(data)
    transformed = ht.transform(data)
    reverse = ht.reverse_transform(transformed)

    # Assert
    pd.testing.assert_frame_equal(reverse, data)


def test_subset_of_columns_nan_data():
    """HyperTransform should be able to transform a subset of the training columns.

    See https://github.com/sdv-dev/RDT/issues/152
    """
    # Setup
    data = get_input_data()
    subset = data[[data.columns[0]]].copy()
    ht = HyperTransformer()
    ht.detect_initial_config(data)
    ht.fit(data)

    # Run
    transformed = ht.transform(subset)
    reverse = ht.reverse_transform(transformed)

    # Assert
    pd.testing.assert_frame_equal(subset, reverse)


def test_multiple_fits():
    """HyperTransformer should be able to be used multiple times.

    Fitting, transforming and reverse transforming should produce the same results when
    called on the same data multiple times.
    """
    # Setup
    data = get_input_data()
    ht = HyperTransformer()

    # Run
    ht.detect_initial_config(data)
    ht.fit(data)
    transformed1 = ht.transform(data)
    reversed1 = ht.reverse_transform(transformed1)

    ht.detect_initial_config(data)
    ht.fit(data)
    transformed2 = ht.transform(data)
    reversed2 = ht.reverse_transform(transformed2)

    # Assert
    pd.testing.assert_frame_equal(transformed1, transformed2)
    pd.testing.assert_frame_equal(reversed1, reversed2)


def test_multiple_fits_different_data():
    """HyperTransformer should be able to be used multiple times regardless of the data.

    Fitting, transforming and reverse transforming should work when called on different data.
    """
    # Setup
    data = pd.DataFrame({'col1': [1, 2, 3], 'col2': [1.0, 0.0, 0.0]})
    new_data = pd.DataFrame({'col2': [1, 2, 3], 'col1': [1.0, 0.0, 0.0]})
    ht = HyperTransformer()

    # Run
    ht.detect_initial_config(data)
    ht.fit(data)
    ht.detect_initial_config(new_data)
    ht.fit(new_data)
    transformed1 = ht.transform(new_data)
    transformed2 = ht.transform(new_data)
    reverse1 = ht.reverse_transform(transformed1)
    reverse2 = ht.reverse_transform(transformed2)

    # Assert
    expected_transformed = pd.DataFrame({'col2.value': [1, 2, 3], 'col1.value': [1.0, 0.0, 0.0]})
    pd.testing.assert_frame_equal(transformed1, expected_transformed)
    pd.testing.assert_frame_equal(transformed2, expected_transformed)
    pd.testing.assert_frame_equal(reverse1, new_data)
    pd.testing.assert_frame_equal(reverse2, new_data)


def test_multiple_fits_different_columns():
    """HyperTransformer should be able to be used multiple times regardless of the data.

    Fitting, transforming and reverse transforming should work when called on different data.
    """
    # Setup
    data = pd.DataFrame({'col1': [1, 2, 3], 'col2': [1.0, 0.0, 0.0]})
    new_data = pd.DataFrame({'col3': [1, 2, 3], 'col4': [1.0, 0.0, 0.0]})
    ht = HyperTransformer()

    # Run
    ht.detect_initial_config(data)
    ht.fit(data)
    ht.detect_initial_config(new_data)
    ht.fit(new_data)
    transformed1 = ht.transform(new_data)
    transformed2 = ht.transform(new_data)
    reverse1 = ht.reverse_transform(transformed1)
    reverse2 = ht.reverse_transform(transformed2)

    # Assert
    expected_transformed = pd.DataFrame({'col3.value': [1, 2, 3], 'col4.value': [1.0, 0.0, 0.0]})
    pd.testing.assert_frame_equal(transformed1, expected_transformed)
    pd.testing.assert_frame_equal(transformed2, expected_transformed)
    pd.testing.assert_frame_equal(reverse1, new_data)
    pd.testing.assert_frame_equal(reverse2, new_data)


def test_multiple_fits_with_set_config():
    """HyperTransformer should be able to be used multiple times regardless of the data.

    Fitting, transforming and reverse transforming should work when called on different data.
    """
    # Setup
    data = get_input_data()
    ht = HyperTransformer()

    # Run
    ht.detect_initial_config(data)
    ht.set_config(config={
        'sdtypes': {'integer': 'float'},
        'transformers': {'bool': FrequencyEncoder}
    })
    ht.fit(data)
    transformed1 = ht.transform(data)
    reverse1 = ht.reverse_transform(transformed1)

    ht.fit(data)
    transformed2 = ht.transform(data)
    reverse2 = ht.reverse_transform(transformed2)

    # Assert
    pd.testing.assert_frame_equal(transformed1, transformed2)
    pd.testing.assert_frame_equal(reverse1, reverse2)


def test_multiple_detect_configs_with_set_config():
    """HyperTransformer should be able to be used multiple times regardless of the data.

    Fitting, transforming and reverse transforming should work when called on different data.
    """
    # Setup
    data = get_input_data()
    ht = HyperTransformer()

    # Run
    ht.detect_initial_config(data)
    ht.fit(data)
    transformed1 = ht.transform(data)
    reverse1 = ht.reverse_transform(transformed1)

    ht.set_config(config={
        'sdtypes': {'integers': 'float'},
        'transformers': {'bool': FrequencyEncoder}
    })

    ht.detect_initial_config(data)
    ht.fit(data)
    transformed2 = ht.transform(data)
    reverse2 = ht.reverse_transform(transformed2)

    # Assert
    pd.testing.assert_frame_equal(transformed1, transformed2)
    pd.testing.assert_frame_equal(reverse1, reverse2)


def test_detect_initial_config_doesnt_affect_fit():
    """HyperTransformer should fit the same way regardless of ``detect_initial_config``.

    Calling the ``detect_initial_config`` method should not affect the results of ``fit``,
    ``transform`` or ``reverse_transform``.
    """
    # Setup
    data = get_input_data()
    ht = HyperTransformer()

    # Run
    ht.detect_initial_config(data)
    ht.fit(data)
    transformed1 = ht.transform(data)
    reversed1 = ht.reverse_transform(transformed1)

    ht.detect_initial_config(data)
    ht.fit(data)
    transformed2 = ht.transform(data)
    reversed2 = ht.reverse_transform(transformed1)

    # Assert
    pd.testing.assert_frame_equal(transformed1, transformed2)
    pd.testing.assert_frame_equal(reversed1, reversed2)


def test_multiple_detects():
    """HyperTransformer should be able to be used multiple times regardless of the data.

    Fitting, transforming and reverse transforming should work when called on different data.
    """
    # Setup
    data = pd.DataFrame({'col2': [1, 2, 3], 'col1': [1.0, 0.0, 0.0]})
    new_data = get_input_data()
    ht = HyperTransformer()

    # Run
    ht.detect_initial_config(data)
    ht.detect_initial_config(new_data)
    ht.fit(new_data)
    transformed = ht.transform(new_data)
    reverse = ht.reverse_transform(transformed)

    # Assert
    pd.testing.assert_frame_equal(transformed, get_transformed_data())
    pd.testing.assert_frame_equal(reverse, get_reversed_data())


def test_transform_without_fit():
    """HyperTransformer should raise an error when transforming without fitting."""
    # Setup
    data = pd.DataFrame()
    ht = HyperTransformer()

    # Run / Assert
    with pytest.raises(NotFittedError):
        ht.transform(data)


def test_fit_data_different_than_detect():
    """HyperTransformer should raise an error when transforming without fitting."""
    # Setup
    ht = HyperTransformer()
    detect_data = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
    data = pd.DataFrame({'col1': [1, 2], 'col3': ['a', 'b']})

    # Run / Assert
    error_msg = re.escape(
        'The data you are trying to fit has different columns than the original '
        "detected data (unknown columns: ['col3']). Column names and their "
        "sdtypes must be the same. Use the method 'get_config()' to see the expected "
        'values.'
    )
    ht.detect_initial_config(detect_data)
    with pytest.raises(NotFittedError, match=error_msg):
        ht.fit(data)
