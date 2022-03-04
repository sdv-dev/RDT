"""Integration tests for the HyperTransformer."""

from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pandas as pd

from rdt import HyperTransformer
from rdt.transformers import (
    DEFAULT_TRANSFORMERS, BaseTransformer, BinaryEncoder, FloatFormatter, FrequencyEncoder,
    OneHotEncoder, UnixTimestampEncoder, get_default_transformer, get_default_transformers)


class DummyTransformerNumerical(BaseTransformer):

    INPUT_TYPE = 'categorical'
    OUTPUT_TYPES = {
        'value': 'float'
    }

    def _fit(self, data):
        pass

    def _transform(self, data):
        return data.astype(float)

    def _reverse_transform(self, data):
        return data.astype(str)


class DummyTransformerNotMLReady(BaseTransformer):

    INPUT_TYPE = 'datetime'
    OUTPUT_TYPES = {
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
        for categorical data types, so that the output is predictable.

    Input:
        - A dataframe with every data type.

    Expected behavior:
        - The transformed data should contain all the ML ready data.
        - The reverse transformed data should be the same as the input.
    """
    # Setup
    data = get_input_data()

    # Run
    ht = HyperTransformer()
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
    in the ``_transform_output_types`` list) should be recursively transformed
    till it is.

    Setup:
        - The datetime column is set to use a dummy transformer that stringifies
        the input. That output is then set to use the categorical transformer.

    Input:
        - A dict mapping each field to a transformer.
        - A dataframe with every data type.

    Expected behavior:
        - The transformed data should contain all the ML ready data.
        - The reverse transformed data should be the same as the input.
    """
    # Setup
    field_transformers = {
        'integer': FloatFormatter(missing_value_replacement='mean'),
        'float': FloatFormatter(missing_value_replacement='mean'),
        'categorical': FrequencyEncoder,
        'bool': BinaryEncoder(missing_value_replacement='mode'),
        'datetime': DummyTransformerNotMLReady,
        'datetime.value': FrequencyEncoder,
        'names': FrequencyEncoder
    }
    data = get_input_data()

    # Run
    ht = HyperTransformer(field_transformers=field_transformers)
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
    ht = HyperTransformer(field_transformers={
        'a': OneHotEncoder()
    })
    data = pd.DataFrame({
        'a': ['a', 'a', 'a']
    })

    # Run
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
    ht.fit(data)

    # Run
    transformed = ht.transform(subset)
    reverse = ht.reverse_transform(transformed)

    # Assert
    pd.testing.assert_frame_equal(subset, reverse)


def test_with_unfitted_columns():
    """HyperTransform should be able to transform even if there are unseen columns in data."""
    # Setup
    data = get_input_data()
    ht = HyperTransformer(default_data_type_transformers={'categorical': FrequencyEncoder})
    ht.fit(data)

    # Run
    new_data = get_input_data()
    new_column = pd.Series([4, 5, 6, 7, 8, 9])
    new_data['z'] = new_column
    transformed = ht.transform(new_data)
    reverse = ht.reverse_transform(transformed)

    # Assert
    expected_reversed = get_reversed_data()
    expected_reversed['z'] = new_column
    expected_reversed = expected_reversed.reindex(
        columns=['z', 'integer', 'float', 'categorical', 'bool', 'datetime', 'names'])
    pd.testing.assert_frame_equal(expected_reversed, reverse)


def test_detect_initial_config_doesnt_affect_fit():
    """HyperTransformer should fit the same way regardless of ``detect_initial_config``.

    Calling the ``detect_initial_config`` method should not affect the results of ``fit``,
    ``transform`` or ``reverse_transform``.
    """
    # Setup
    data = get_input_data()
    ht = HyperTransformer()

    # Run
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


def test_detect_initial_config():
    """HyperTransformer should reset its state when ``detect_initial_config`` runs."""
    # Setup
    data = pd.DataFrame({'col': ['a', 'b', 'c'], 'col2': [1, 2, 3]})
    new_data = pd.DataFrame({'col': [1, 2, 3], 'col3': ['a', 'b', 'c']})
    ht = HyperTransformer()
    ht.fit(data)

    # Run
    ht.detect_initial_config(data)
    sdtypes1 = ht.field_data_types
    transformers1 = {k: repr(v) for (k, v) in ht.field_transformers.items()}

    ht.fit(data)
    sdtypes2 = ht.field_data_types
    transformers2 = {k: repr(v) for (k, v) in ht.field_transformers.items()}

    ht.detect_initial_config(new_data)
    sdtypes3 = ht.field_data_types
    transformers3 = {k: repr(v) for (k, v) in ht.field_transformers.items()}

    ht.fit(new_data)
    sdtypes4 = ht.field_data_types
    transformers4 = {k: repr(v) for (k, v) in ht.field_transformers.items()}

    # Assert
    assert sdtypes1 == sdtypes2 == {'col': 'categorical', 'col2': 'integer'}
    assert transformers1 == transformers2 == {
        'col': 'FrequencyEncoder()',
        'col2': "FloatFormatter(missing_value_replacement='mean')"
    }

    assert sdtypes3 == sdtypes4 == {'col': 'integer', 'col3': 'categorical'}
    assert transformers3 == transformers4 == {
        'col': "FloatFormatter(missing_value_replacement='mean')",
        'col3': 'FrequencyEncoder()'
    }
