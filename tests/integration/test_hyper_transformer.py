"""Integration tests for the HyperTransformer."""

from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pandas as pd

from rdt import HyperTransformer
from rdt.transformers import (
    DEFAULT_TRANSFORMERS, BaseTransformer, BooleanTransformer, CategoricalTransformer,
    DatetimeTransformer, NumericalTransformer, OneHotEncodingTransformer)


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


class DummyTransformerMultiColumn(BaseTransformer):

    INPUT_TYPE = 'datetime'
    OUTPUT_TYPES = {
        'value': 'float',
    }

    def _fit(self, data):
        pass

    def _transform(self, data):
        # Convert multiple columns into a single datetime
        data.columns = [c.replace('_str.value', '') for c in data.columns]
        data = pd.to_datetime(data)

        float_data = data.to_numpy().astype(np.float64)
        data_is_nan = data.isna().to_numpy().astype(np.float64)

        output = dict(zip(
            self.output_columns,
            [float_data, data_is_nan]
        ))

        output = pd.DataFrame(output).fillna(-1)

        return output

    def _reverse_transform(self, data):
        datetimes = data.round().astype('datetime64[ns]')
        out = pd.DataFrame({
            'year': datetimes.dt.year,
            'month': datetimes.dt.month,
            'day': datetimes.dt.day,
        })

        return out


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
        'float.is_null': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        'categorical.value': [0.3125, 0.3125, 0.9375, 0.75, 0.3125, 0.75, 0.3125, 0.3125],
        'bool.value': [0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        'bool.is_null': [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'datetime.value': datetimes,
        'datetime.is_null': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'names.value': [0.3125, 0.75, 0.75, 0.3125, 0.3125, 0.9375, 0.3125, 0.3125]
    }, index=TEST_DATA_INDEX)


DETERMINISTIC_DEFAULT_TRANSFORMERS = deepcopy(DEFAULT_TRANSFORMERS)
DETERMINISTIC_DEFAULT_TRANSFORMERS['categorical'] = CategoricalTransformer


@patch('rdt.transformers.DEFAULT_TRANSFORMERS', DETERMINISTIC_DEFAULT_TRANSFORMERS)
def test_hypertransformer_default_inputs():
    """Test the HyperTransformer with default parameters.

    This tests that if default parameters are provided to the HyperTransformer,
    the ``default_transformers`` method will be used to determine which
    transformers to use for each field.

    Setup:
        - Patch the DEFAULT_TRANSFORMERS to use the `CategoricalTransformer`
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

    expected_reversed = get_input_data()
    pd.testing.assert_frame_equal(expected_reversed, reverse_transformed)

    assert isinstance(ht._transformers_tree['integer']['transformer'], NumericalTransformer)
    assert ht._transformers_tree['integer']['outputs'] == ['integer.value']
    assert isinstance(ht._transformers_tree['float']['transformer'], NumericalTransformer)
    assert ht._transformers_tree['float']['outputs'] == ['float.value', 'float.is_null']
    assert isinstance(ht._transformers_tree['categorical']['transformer'], CategoricalTransformer)
    assert ht._transformers_tree['categorical']['outputs'] == ['categorical.value']
    assert isinstance(ht._transformers_tree['bool']['transformer'], BooleanTransformer)
    assert ht._transformers_tree['bool']['outputs'] == ['bool.value', 'bool.is_null']
    assert isinstance(ht._transformers_tree['datetime']['transformer'], DatetimeTransformer)
    assert ht._transformers_tree['datetime']['outputs'] == ['datetime.value', 'datetime.is_null']
    assert isinstance(ht._transformers_tree['names']['transformer'], CategoricalTransformer)
    assert ht._transformers_tree['names']['outputs'] == ['names.value']


def test_hypertransformer_field_transformers():
    """Test the HyperTransformer with ``field_transformers`` provided.

    This tests that this transformers specified in the ``field_transformers``
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
        'integer': NumericalTransformer(dtype=np.int64),
        'float': NumericalTransformer(dtype=float),
        'categorical': CategoricalTransformer,
        'bool': BooleanTransformer,
        'datetime': DummyTransformerNotMLReady,
        'datetime.value': CategoricalTransformer,
        'names': CategoricalTransformer
    }
    data = get_input_data()

    # Run
    ht = HyperTransformer(field_transformers=field_transformers)
    ht.fit(data)
    transformed = ht.transform(data)
    reverse_transformed = ht.reverse_transform(transformed)

    # Assert
    expected_transformed = get_transformed_data()
    del expected_transformed['datetime.is_null']
    rename = {'datetime.value': 'datetime.value.value'}
    expected_transformed = expected_transformed.rename(columns=rename)
    transformed_datetimes = [0.9375, 0.75, 0.3125, 0.3125, 0.3125, 0.75, 0.3125, 0.3125]
    expected_transformed['datetime.value.value'] = transformed_datetimes
    pd.testing.assert_frame_equal(transformed, expected_transformed)

    expected_reversed = get_input_data()
    pd.testing.assert_frame_equal(expected_reversed, reverse_transformed)


def test_hypertransformer_field_transformers_multi_column_fields():
    """Test the HyperTransformer with ``field_transformers`` provided.

    This test will make sure that fields made up of multiple columns are
    properly handled if they are specified in the ``field_transformers``
    argument. If the field has one column that is derived from another
    transformer, then the other transformer should be transformed and
    the multi-column field should be handled when all its columns are
    present.

    Setup:
        - A dummy transformer that takes in a column for day, year and
        month and creates one numeric value from it.
        - A dummy transformer that takes a string value and parses the
        float from it.

    Input:
        - A dict mapping each field to a dummy transformer.
        - A dataframe with a numerical year, month and day column as well
        as a string year, month and day column.

    Expected behavior:
        - The transformed data should contain all the ML ready data.
        - The reverse transformed data should be the same as the input.
    """
    # Setup
    data = pd.DataFrame({
        'year': [2001, 2002, 2003],
        'month': [1, 2, 3],
        'day': [1, 2, 3],
        'year_str': ['2001', '2002', '2003'],
        'month_str': ['1', '2', '3'],
        'day_str': ['1', '2', '3'],
    })
    field_transformers = {
        ('year', 'month', 'day'): DummyTransformerMultiColumn,
        'year_str': DummyTransformerNumerical,
        'day_str': DummyTransformerNumerical,
        'month_str': DummyTransformerNumerical,
        ('year_str.value', 'month_str.value', 'day_str.value'): DummyTransformerMultiColumn
    }
    expected_transformed = pd.DataFrame({
        'year#month#day.value': [
            9.783072e+17,
            1.012608e+18,
            1.046650e+18
        ],
        'year_str.value#month_str.value#day_str.value.value': [
            9.783072e+17,
            1.012608e+18,
            1.046650e+18
        ]
    })
    expected_reversed = data.copy()

    # Run
    ht = HyperTransformer(field_transformers=field_transformers)
    ht.fit(data)
    transformed = ht.transform(data)
    reverse_transformed = ht.reverse_transform(transformed)

    # Assert
    pd.testing.assert_frame_equal(transformed, expected_transformed)
    pd.testing.assert_frame_equal(expected_reversed, reverse_transformed)


def test_single_category():
    """Test that categorical variables with a single value are supported."""
    # Setup
    ht = HyperTransformer(field_transformers={
        'a': OneHotEncodingTransformer()
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
    ht = HyperTransformer(default_data_type_transformers={'categorical': CategoricalTransformer})
    ht.fit(data)

    # Run
    new_data = get_input_data()
    new_column = pd.Series([4, 5, 6, 7, 8, 9])
    new_data['z'] = new_column
    transformed = ht.transform(new_data)
    reverse = ht.reverse_transform(transformed)

    # Assert
    expected_reversed = get_input_data()
    expected_reversed['z'] = new_column
    expected_reversed = expected_reversed.reindex(
        columns=['z', 'integer', 'float', 'categorical', 'bool', 'datetime', 'names'])
    pd.testing.assert_frame_equal(expected_reversed, reverse)
