import random

import numpy as np
import pandas as pd
import pytest

from rdt import HyperTransformer
from rdt.transformers import (
    BaseTransformer, BooleanTransformer, CategoricalTransformer, NumericalTransformer,
    OneHotEncodingTransformer)


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

        output = dict(zip(
            self.output_columns,
            [data.values.astype(np.float64), data.isnull().astype(np.float64)]
        ))

        output = pd.DataFrame(output)
        output.fillna(-1, inplace=True)

        return output

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


def get_input_data_without_nan(index=None):
    data = pd.DataFrame({
        'integer': [1, 2, 1, 3],
        'float': [0.1, 0.2, 0.1, 0.1],
        'categorical': ['a', 'a', 'b', 'a'],
        'bool': [False, False, True, False],
        'datetime': ['2010-02-01', '2010-01-01', '2010-02-01', '2010-02-01'],
        'names': ['Jon', 'Arya', 'Jon', 'Jon'],
    })
    data['datetime'] = pd.to_datetime(data['datetime'])
    if index:
        data.index = index

    return data


def get_reversed(index=None):
    reverse_transformed = get_input_data_without_nan(index)
    reverse_transformed['bool'] = reverse_transformed['bool'].astype('O')
    return reverse_transformed


def get_transformed_data(index=None):
    transformed = pd.DataFrame({
        'integer.value': [1, 2, 1, 3],
        'float.value': [0.1, 0.2, 0.1, 0.1],
        'categorical.value': [0.375, 0.375, 0.875, 0.375],
        'bool.value': [0.0, 0.0, 1.0, 0.0],
        'datetime.value': [
            1.2649824e+18,
            1.262304e+18,
            1.2649824e+18,
            1.2649824e+18
        ],
        'names.value': [0.375, 0.875, 0.375, 0.375]
    })
    if index:
        transformed.index = index

    return transformed


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


def test_hypertransformer_default_inputs():
    """Test the HyperTransformer with default parameters.

<<<<<<< HEAD
    ht = HyperTransformer(transformers)
    ht.fit(data)
    transformed = ht.transform(data)

    expected = get_transformed_data()

    np.testing.assert_allclose(
        transformed.sort_index(axis=1).to_numpy(),
        expected.sort_index(axis=1).to_numpy()
    )

    reversed_data = ht.reverse_transform(transformed)
=======
    This tests that if default parameters are provided to the HyperTransformer,
    the ``default_transformers`` method will be used to determine which
    transformers to use for each field.
>>>>>>> v0.6.0-dev

    Setup:
        - `data_type_transformers` will be set to use the `CategoricalTransformer`
        for categorical data types so that the output is predictable.

    Input:
        - A dataframe with every data type.

    Expected behavior:
        - The transformed data should contain all the ML ready data.
        - The reverse transformed data should be the same as the input.
    """
    index = random.shuffle(list(range(4)))
    data = get_input_data_without_nan(index)
    expected_transformed = data.drop('datetime', axis=1)
    expected_transformed.columns = [
        'integer.value',
        'float.value',
        'categorical.value',
        'bool.value',
        'names.value'
    ]
    expected_transformed = get_transformed_data(index)
    expected_reversed = get_reversed(index)

    ht = HyperTransformer(data_type_transformers={'categorical': CategoricalTransformer})
    ht.fit(data)
    transformed = ht.transform(data)

    pd.testing.assert_frame_equal(transformed, expected_transformed)

    reverse_transformed = ht.reverse_transform(transformed)

    pd.testing.assert_frame_equal(expected_reversed, reverse_transformed)


def test_hypertransformer_field_transformers():
    """Test the HyperTransformer with ``field_transformers`` provided.

    This tests that this transformers specified in the ``field_transformers``
    argument are used. Any output of a transformer that is not ML ready (not
    in the ``_transform_output_types`` list) should be recursively transformed
    till it is.

    Setup:
        - The datetime column is set to use a dumm transformer that stringifies
        the input. That output is then set to use the categorical transformer.

    Input:
        - A dict mapping each field to a transformer.
        - A dataframe with every data type.

    Expected behavior:
        - The transformed data should contain all the ML ready data.
        - The reverse transformed data should be the same as the input.
    """
    field_transformers = {
        'integer': NumericalTransformer(dtype=np.int64),
        'float': NumericalTransformer(dtype=float),
        'categorical': CategoricalTransformer,
        'bool': BooleanTransformer,
        'datetime': DummyTransformerNotMLReady,
        'datetime.value': CategoricalTransformer,
        'names': CategoricalTransformer
    }
    data = get_input_data_without_nan()
    expected_transformed = get_transformed_data().rename(
        columns={'datetime.value': 'datetime.value.value'})
    expected_transformed['datetime.value.value'] = [0.375, 0.875, 0.375, 0.375]
    expected_reversed = get_reversed()

    ht = HyperTransformer(field_transformers=field_transformers)
    ht.fit(data)
    transformed = ht.transform(data)

    pd.testing.assert_frame_equal(
        transformed,
        expected_transformed
    )

    reverse_transformed = ht.reverse_transform(transformed)

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

    ht = HyperTransformer(field_transformers=field_transformers)
    ht.fit(data)
    transformed = ht.transform(data)

    pd.testing.assert_frame_equal(
        transformed,
        expected_transformed
    )

    reverse_transformed = ht.reverse_transform(transformed)

    pd.testing.assert_frame_equal(expected_reversed, reverse_transformed)


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


@pytest.mark.xfail()
def test_dtype_category():
    data = pd.DataFrame({'a': ['a', 'b', 'c']}, dtype='category')

    ht = HyperTransformer()
    ht.fit(data)

    trans = ht.transform(data)

    rever = ht.reverse_transform(trans)

    pd.testing.assert_frame_equal(rever, data)


def test_subset_of_columns():
    """HyperTransform should be able to transform a subset of the training columns.

    See https://github.com/sdv-dev/RDT/issues/152
    """
    data = get_input_data_without_nan()

    ht = HyperTransformer()
    ht.fit(data)

    subset = get_input_data_without_nan()[[data.columns[0]]]
    transformed = ht.transform(subset)
    reverse = ht.reverse_transform(transformed)

    pd.testing.assert_frame_equal(subset, reverse)


def test_with_unfitted_columns():
    """HyperTransform should be able to transform even if there are unseen columns in data."""
    data = get_input_data_without_nan()

    ht = HyperTransformer(data_type_transformers={'categorical': CategoricalTransformer})
    ht.fit(data)

    new_data = get_input_data_without_nan()
    new_column = pd.Series([6, 7, 8, 9])
    new_data['z'] = new_column
    transformed = ht.transform(new_data)
    reverse = ht.reverse_transform(transformed)

    expected_reversed = get_reversed()
    expected_reversed['z'] = new_column
    expected_reversed = expected_reversed.reindex(
        columns=['z', 'integer', 'float', 'categorical', 'bool', 'datetime', 'names'])
    pd.testing.assert_frame_equal(expected_reversed, reverse)


def test_subset_of_columns_nan_data():
    """HyperTransform should be able to transform a subset of the training columns.

    See https://github.com/sdv-dev/RDT/issues/152
    """
    data = get_input_data_with_nan()

    ht = HyperTransformer()
    ht.fit(data)

    subset = get_reversed()[[data.columns[0]]]
    transformed = ht.transform(subset)
    reverse = ht.reverse_transform(transformed)

    pd.testing.assert_frame_equal(subset, reverse)
