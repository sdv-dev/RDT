from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from rdt import HyperTransformer
from rdt.transformers import BaseTransformer, OneHotEncodingTransformer, get_default_transformer


class DummyTransformer(BaseTransformer):

    INPUT_TYPE = 'any'
    OUTPUT_TYPES = {
        'value': 'float'
    }

    def _fit(self, data):
        pass

    def _transform(self, data):
        return data

    def _reverse_transform(self, data):
        return data


class DummyTransformerNumerical(BaseTransformer):

    INPUT_TYPE = 'any'
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

        out = pd.DataFrame(dict(zip(
            self.output_columns,
            [
                data.values.astype(np.float64),
                data.isnull().astype(np.float64)
            ]
        ))).fillna(-1)

        return out

    def _reverse_transform(self, data):
        datetimes = data.round().astype('datetime64[ns]')
        out = pd.DataFrame({
            'year': datetimes.dt.year,
            'month': datetimes.dt.month,
            'day': datetimes.dt.day,
        })

        return out


def get_input_data_with_nan():
    data = pd.DataFrame({
        'integer': [1, 2, 1, 3, 1],
        'float': [0.1, 0.2, 0.1, np.nan, 0.1],
        'categorical': ['a', 'a', np.nan, 'b', 'a'],
        'bool': [False, np.nan, False, True, False],
        'datetime': [
            np.nan, '2010-02-01', '2010-01-01', '2010-02-01', '2010-01-01'
        ],
        'names': ['Jon', 'Arya', 'Arya', 'Jon', 'Jon'],
    })
    data['datetime'] = pd.to_datetime(data['datetime'])

    return data


def get_input_data_without_nan():
    data = pd.DataFrame({
        'integer': [1, 2, 1, 3],
        'float': [0.1, 0.2, 0.1, 0.1],
        'categorical': ['a', 'a', 'b', 'a'],
        'bool': [False, False, True, False],
        'datetime': [
            '2010-02-01', '2010-01-01', '2010-02-01', '2010-01-01'
        ],
        'names': ['Jon', 'Arya', 'Jon', 'Jon'],
    })
    data['datetime'] = pd.to_datetime(data['datetime'])
    data['bool'] = data['bool'].astype('O')  # boolean transformer returns O instead of bool

    return data


def get_transformed_data():
    return pd.DataFrame({
        'integer': [1, 2, 1, 3],
        'float': [0.1, 0.2, 0.1, 0.1],
        'categorical': [0.375, 0.375, 0.875, 0.375],
        'bool': [0.0, 0.0, 1.0, 0.0],
        'datetime': [
            1.2649824e+18,
            1.262304e+18,
            1.2649824e+18,
            1.262304e+18
        ],
        'names': [0.375, 0.875, 0.375, 0.375]
    })


def get_transformed_nan_data():
    return pd.DataFrame({
        'integer': [1, 2, 1, 3, 1],
        'float': [0.1, 0.2, 0.1, 0.125, 0.1],
        'float#1': [0.0, 0.0, 0.0, 1.0, 0.0],
        'categorical': [0.3, 0.3, 0.9, 0.7, 0.3],
        'bool': [0.0, -1.0, 0.0, 1.0, 0.0],
        'bool#1': [0.0, 1.0, 0.0, 0.0, 0.0],
        'datetime': [
            1.2636432e+18, 1.2649824e+18, 1.262304e+18,
            1.2649824e+18, 1.262304e+18
        ],
        'datetime#1': [1.0, 0.0, 0.0, 0.0, 0.0],
        'names': [0.3, 0.8, 0.8, 0.3, 0.3],
    })


def get_transformers():
    return {
        'integer': {
            'class': 'NumericalTransformer',
            'kwargs': {
                'dtype': np.int64,
            }
        },
        'float': {
            'class': 'NumericalTransformer',
            'kwargs': {
                'dtype': np.float64,
            }
        },
        'categorical': {
            'class': 'CategoricalTransformer'
        },
        'bool': {
            'class': 'BooleanTransformer'
        },
        'datetime': {
            'class': 'DatetimeTransformer'
        },
        'names': {
            'class': 'CategoricalTransformer',
        },
    }


@patch('rdt.transformers.get_default_transformers')
def test_hypertransformer_no_inputs(default_transformers_mock):
    """Test the HyperTransformer with no transformers provided.

    This tests that if no parameters are provided to the HyperTransformer,
    the ``default_transformers`` method will be used to determine which
    transformers to use for each field. Any output of a transformer that
    is not ML ready should be recursively transformed till it is.

    Setup:
        - A dummy transformer that just returns the input data.
        - Another dummy transformer that returns data that is not ML ready.
        - A dummy dict that replaces the ``default_transformers`` method
        so that all the field types use the dummy transformers created.

    Input:
        - A dataframe with every data type.

    Expected behavior:
        - The transformed data should contain all the ML ready data.
        - The reverse transformed data should be the same as the input.
    """
    get_default_transformer.cache_clear()
    default_transformers_mock.return_value = {
        'numerical': DummyTransformer,
        'integer': DummyTransformer,
        'float': DummyTransformer,
        'categorical': DummyTransformer,
        'boolean': DummyTransformer,
        'datetime': DummyTransformerNotMLReady,
    }
    data = get_input_data_without_nan()
    expected_transformed = data.drop('datetime', axis=1)
    expected_transformed.columns = [
        'integer.value',
        'float.value',
        'categorical.value',
        'bool.value',
        'names.value'
    ]
    expected_transformed['datetime.value.value'] = [
        '2010-02-01', '2010-01-01', '2010-02-01', '2010-01-01'
    ]
    expected_reversed = data.copy()

    ht = HyperTransformer()
    ht.fit(data)
    transformed = ht.transform(data)

    pd.testing.assert_frame_equal(
        transformed.sort_index(axis=1),
        expected_transformed.sort_index(axis=1)
    )

    reversed = ht.reverse_transform(transformed)

    pd.testing.assert_frame_equal(
        expected_reversed.sort_index(axis=1),
        reversed.sort_index(axis=1)
    )


def test_hypertransformer_field_transformers():
    """Test the HyperTransformer with ``field_transformers`` provided.

    This tests that this transformers specified in the ``field_transformers``
    argument are used. Any output of a transformer that is not ML ready should
    be recursively transformed till it is.

    Setup:
        - A dummy transformer that just returns the input data.
        - Another dummy transformer that returns data that is not ML ready.

    Input:
        - A dict mapping each field to a dummy transformer.
        - A dataframe with every data type.

    Expected behavior:
        - The transformed data should contain all the ML ready data.
        - The reverse transformed data should be the same as the input.
    """
    field_transformers = {
        'integer': DummyTransformer,
        'float': DummyTransformer,
        'categorical': DummyTransformer,
        'bool': DummyTransformer,
        'datetime': DummyTransformerNotMLReady,
        'datetime.value': DummyTransformer,
        'names': DummyTransformer
    }
    data = get_input_data_without_nan()
    expected_transformed = data.drop('datetime', axis=1)
    expected_transformed.columns = [
        'integer.value',
        'float.value',
        'categorical.value',
        'bool.value',
        'names.value'
    ]
    expected_transformed['datetime.value.value'] = [
        '2010-02-01', '2010-01-01', '2010-02-01', '2010-01-01'
    ]
    expected_reversed = data.copy()

    ht = HyperTransformer(field_transformers=field_transformers)
    ht.fit(data)
    transformed = ht.transform(data)

    pd.testing.assert_frame_equal(
        transformed.sort_index(axis=1),
        expected_transformed.sort_index(axis=1)
    )

    reversed = ht.reverse_transform(transformed)

    pd.testing.assert_frame_equal(
        expected_reversed.sort_index(axis=1),
        reversed.sort_index(axis=1)
    )


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
        - A dataframe with a nuerical year, month and day column as well
        as a string year, month and day column.

    Expected behavior:
        - The transformed data should contain all the ML ready data.
        - The reverse transformed data should be the same as the input.
    """
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
        'year#month#day.value': [9.783072e+17, 1.012608e+18, 1.046650e+18],
        'year_str.value#month_str.value#day_str.value.value': [
            9.783072e+17, 1.012608e+18, 1.046650e+18]
    })
    expected_reversed = data.copy()

    ht = HyperTransformer(field_transformers=field_transformers)
    ht.fit(data)
    transformed = ht.transform(data)

    pd.testing.assert_frame_equal(
        transformed.sort_index(axis=1),
        expected_transformed.sort_index(axis=1)
    )

    reversed = ht.reverse_transform(transformed)

    pd.testing.assert_frame_equal(
        expected_reversed.sort_index(axis=1),
        reversed.sort_index(axis=1)
    )


def test_single_category():
    ht = HyperTransformer(field_transformers={
        'a': OneHotEncodingTransformer()
    })
    data = pd.DataFrame({
        'a': ['a', 'a', 'a']
    })

    ht.fit(data)
    transformed = ht.transform(data)

    reverse = ht.reverse_transform(transformed)

    pd.testing.assert_frame_equal(data, reverse)


@pytest.mark.xfail
def test_dtype_category():
    df = pd.DataFrame({'a': ['a', 'b', 'c']}, dtype='category')

    ht = HyperTransformer()
    ht.fit(df)

    trans = ht.transform(df)

    rever = ht.reverse_transform(trans)

    pd.testing.assert_frame_equal(rever, df)


def test_subset_of_columns():
    """HyperTransform should be able to transform a subset of the training columns.

    See https://github.com/sdv-dev/RDT/issues/152
    """
    data = get_input_data_without_nan()

    ht = HyperTransformer()
    ht.fit(data)

    subset = data[[data.columns[0]]]
    transformed = ht.transform(subset)
    reverse = ht.reverse_transform(transformed)

    pd.testing.assert_frame_equal(subset, reverse)


def test_subset_of_columns_nan_data():
    """HyperTransform should be able to transform a subset of the training columns.

    See https://github.com/sdv-dev/RDT/issues/152
    """
    data = get_input_data_with_nan()

    ht = HyperTransformer()
    ht.fit(data)

    subset = data[[data.columns[0]]]
    transformed = ht.transform(subset)
    reverse = ht.reverse_transform(transformed)

    pd.testing.assert_frame_equal(subset, reverse)
