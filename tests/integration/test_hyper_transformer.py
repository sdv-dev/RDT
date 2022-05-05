"""Integration tests for the HyperTransformer."""

import re
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from rdt import HyperTransformer
from rdt.errors import Error, NotFittedError
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
        '2010-02-01',
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
        'float': [0.1, 0.2, 0.1, 0.2, 0.1, 0.4, 0.2, 0.3],
        'categorical': ['a', 'a', 'b', 'b', 'a', 'b', 'a', 'a'],
        'bool': [False, False, False, True, False, False, True, False],
        'datetime': datetimes,
        'names': ['Jon', 'Arya', 'Arya', 'Jon', 'Jon', 'Sansa', 'Jon', 'Jon'],
    }, index=TEST_DATA_INDEX)

    return data


def get_transformed_data():
    datetimes = [
        1.264982e+18,
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
        'categorical.value': [0.3125, 0.3125, .8125, 0.8125, 0.3125, 0.8125, 0.3125, 0.3125],
        'bool.value': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        'datetime.value': datetimes,
        'names.value': [0.3125, 0.75, 0.75, 0.3125, 0.3125, 0.9375, 0.3125, 0.3125]
    }, index=TEST_DATA_INDEX)


def get_reversed_data():
    data = get_input_data()
    data['bool'] = data['bool'].astype('object')
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
        - A fixed random seed to guarantee the samle values are null.

    Expected behavior:
        - The transformed data should contain all the ML ready data.
        - The reverse transformed data should be the same as the input.
    """
    # Setup
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
        'bool': [False, np.nan, False, True, False, np.nan, True, False],
        'datetime': datetimes,
        'names': ['Jon', 'Arya', 'Arya', 'Jon', 'Jon', 'Sansa', 'Jon', 'Jon'],
    }, index=TEST_DATA_INDEX)

    # Run
    ht = HyperTransformer()
    ht.detect_initial_config(data)
    ht.fit(data)
    transformed = ht.transform(data)
    reverse_transformed = ht.reverse_transform(transformed)

    # Assert
    expected_datetimes = [
        1.263069e+18,
        1.264982e+18,
        1.262304e+18,
        1.262304e+18,
        1.262304e+18,
        1.264982e+18,
        1.262304e+18,
        1.262304e+18
    ]
    expected_transformed = pd.DataFrame({
        'integer.value': [1, 2, 1, 3, 1, 4, 2, 3],
        'float.value': [0.1, 0.2, 0.1, 0.2, 0.1, 0.4, 0.2, 0.3],
        'categorical.value': [0.3125, 0.3125, 0.9375, 0.75, 0.3125, 0.75, 0.3125, 0.3125],
        'bool.value': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        'datetime.value': expected_datetimes,
        'names.value': [0.3125, 0.75, 0.75, 0.3125, 0.3125, 0.9375, 0.3125, 0.3125]
    }, index=TEST_DATA_INDEX)
    pd.testing.assert_frame_equal(transformed, expected_transformed)

    reversed_datetimes = pd.to_datetime([
        '2010-01-09 20:34:17.142857216',
        '2010-02-01',
        '2010-01-01',
        '2010-01-01',
        '2010-01-01',
        '2010-02-01',
        '2010-01-01',
        '2010-01-01',
    ])
    expected_reversed = pd.DataFrame({
        'integer': [1, 2, 1, 3, 1, 4, 2, 3],
        'float': [0.1, 0.2, 0.1, 0.20000000000000004, 0.1, 0.4, 0.20000000000000004, 0.3],
        'categorical': ['a', 'a', np.nan, 'b', 'a', 'b', 'a', 'a'],
        'bool': [False, False, False, True, False, False, True, False],
        'datetime': reversed_datetimes,
        'names': ['Jon', 'Arya', 'Arya', 'Jon', 'Jon', 'Sansa', 'Jon', 'Jon'],
    }, index=TEST_DATA_INDEX)
    for row in range(reverse_transformed.shape[0]):
        for column in range(reverse_transformed.shape[1]):
            expected = expected_reversed.iloc[row, column]
            actual = reverse_transformed.iloc[row, column]
            assert pd.isna(actual) or expected == actual

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
        'sdtypes': {
            'integer': 'numerical',
            'float': 'numerical',
            'categorical': 'categorical',
            'bool': 'boolean',
            'datetime': 'datetime',
            'names': 'categorical'
        },
        'transformers': {
            'integer': FloatFormatter(missing_value_replacement='mean'),
            'float': FloatFormatter(missing_value_replacement='mean'),
            'categorical': FrequencyEncoder,
            'bool': BinaryEncoder(missing_value_replacement='mode'),
            'datetime': DummyTransformerNotMLReady,
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
    transformed_datetimes = [0.8125, 0.8125, 0.3125, 0.3125, 0.3125, 0.8125, 0.3125, 0.3125]
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
        'sdtypes': {'integer': 'categorical'},
        'transformers': {'integer': FrequencyEncoder}
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
        'sdtypes': {'integers': 'categorical'},
        'transformers': {'integers': FrequencyEncoder}
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
    ht.detect_initial_config(data)

    # Run / Assert
    with pytest.raises(Error):
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
    with pytest.raises(Error, match=error_msg):
        ht.fit(data)


def test_transform_without_fitting():
    """HyperTransformer shouldn't transform when fit hasn't been called yet."""
    # Setup
    data = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
    ht = HyperTransformer()

    # Run / Assert
    ht.detect_initial_config(data)
    error_msg = (
        'The HyperTransformer is not ready to use. Please fit your data first using '
        "'fit' or 'fit_transform'."
    )
    with pytest.raises(NotFittedError, match=error_msg):
        ht.transform(data)


def test_transform_without_refitting():
    """HyperTransformer shouldn't transform when a new config hasn't been fitted."""
    # Setup
    data = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
    ht = HyperTransformer()

    # Run / Assert
    ht.detect_initial_config(data)
    ht.fit(data)
    ht.update_sdtypes({'col1': 'categorical'})
    error_msg = (
        'The HyperTransformer is not ready to use. Please fit your data first using '
        "'fit' or 'fit_transform'."
    )
    with pytest.raises(NotFittedError, match=error_msg):
        ht.transform(data)


def test_transform_without_config():
    """HyperTransformer shouldn't transform when a config hasn't been set."""
    # Setup
    data = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
    ht = HyperTransformer()

    # Run / Assert
    error_msg = (
        "No config detected. Set the config using 'set_config' or pre-populate "
        "it automatically from your data using 'detect_initial_config' prior to "
        'fitting your data.'
    )
    with pytest.raises(Error, match=error_msg):
        ht.transform(data)


def test_transform_unseen_columns():
    """HyperTransformer shouldn't transform when the data wasn't seen during fit."""
    # Setup
    data = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
    different_data = pd.DataFrame({'col3': [1, 2]})
    ht = HyperTransformer()

    # Run / Assert
    ht.detect_initial_config(data)
    ht.fit(data)
    error_msg = error_msg = (
        'The data you are trying to transform has different columns than the original data. '
        'Column names and their sdtypes must be the same.'
    )
    with pytest.raises(Error, match=error_msg):
        ht.transform(different_data)


def test_update_sdtypes_incorrect_columns():
    """HyperTransformer should crash when update_sdytpes is passed non-existing columns."""
    # Setup
    data = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
    column_name_to_sdtype = {'col3': [1, 2]}
    ht = HyperTransformer()

    # Run / Assert
    ht.detect_initial_config(data)
    error_msg = error_msg = re.escape(
        "Invalid column names: ['col3']. These columns do not exist in the "
        "config. Use 'set_config()' to write and set your entire config at once."
    )
    with pytest.raises(Error, match=error_msg):
        ht.update_sdtypes(column_name_to_sdtype)


def test_update_sdtypes_incorrect_sdtype():
    """HyperTransformer should crash when update_sdytpes is passed non-existing columns."""
    # Setup
    data = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
    column_name_to_sdtype = {'col1': 'unexpected'}
    ht = HyperTransformer()

    # Run / Assert
    ht.detect_initial_config(data)
    error_msg = error_msg = re.escape(
        "Invalid sdtypes: ['unexpected']. If you are trying to use a "
        'premium sdtype, contact info@sdv.dev about RDT Add-Ons.'
    )
    with pytest.raises(Error, match=error_msg):
        ht.update_sdtypes(column_name_to_sdtype)


def test_transform_subset():
    """Test the ``transform_subset`` method.

    The method should return a ``pandas.DataFrame`` with the subset of columns transformed.

    Setup:
        - Detect the config and fit the data.

    Input:
        - A ``pandas.DataFrame`` with a subset of the fitted columns.

    Ouput:
        - A ``pandas.DataFrame`` with the subset transformed.
    """
    # Setup
    data = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
    subset = data[['col1']]
    ht = HyperTransformer()
    ht.detect_initial_config(data)
    ht.fit(data)

    # Run
    transformed = ht.transform_subset(subset)

    # Assert
    expected = pd.DataFrame({'col1.value': [1, 2]})
    pd.testing.assert_frame_equal(transformed, expected)


def test_reverse_transform_subset():
    """Test the ``reverse_transform_subset`` method.

    The method should return a ``pandas.DataFrame`` with the subset of columns reverse transformed.

    Setup:
        - Detect the config and fit the data.

    Input:
        - A ``pandas.DataFrame`` with a subset of the output columns.

    Ouput:
        - A ``pandas.DataFrame`` with the subset reverse transformed.
    """
    # Setup
    data = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
    subset = pd.DataFrame({'col1.value': [1, 2]})
    ht = HyperTransformer()
    ht.detect_initial_config(data)
    ht.fit(data)

    # Run
    reverse_transformed = ht.reverse_transform_subset(subset)

    # Assert
    expected = pd.DataFrame({'col1': [1, 2]})
    pd.testing.assert_frame_equal(reverse_transformed, expected)
