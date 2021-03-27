import numpy as np
import pandas as pd

from rdt import HyperTransformer
from rdt.transformers import OneHotEncodingTransformer


def get_input_data_with_nan():
    data = pd.DataFrame({
        'integer': [1, 2, 1, 3, 1],
        'float': [0.1, 0.2, 0.1, np.nan, 0.1],
        'categorical': ['a', 'b', np.nan, 'b', 'a'],
        'bool': [False, np.nan, False, True, False],
        'datetime': [
            np.nan, '2010-02-01', '2010-01-01', '2010-02-01', '2010-01-01'
        ],
        'names': ['Jon', 'Arya', 'Sansa', 'Jon', 'Robb'],
    })
    data['datetime'] = pd.to_datetime(data['datetime'])

    return data


def get_input_data_without_nan():
    data = pd.DataFrame({
        'integer': [1, 2, 1, 3],
        'float': [0.1, 0.2, 0.1, 0.1],
        'categorical': ['a', 'b', 'b', 'a'],
        'bool': [False, False, True, False],
        'datetime': [
            '2010-02-01', '2010-01-01', '2010-02-01', '2010-01-01'
        ],
        'names': ['Jon', 'Arya', 'Sansa', 'Jon'],
    })
    data['datetime'] = pd.to_datetime(data['datetime'])
    data['bool'] = data['bool'].astype('O')  # boolean transformer returns O instead of bool

    return data


def get_transformed_data():
    return pd.DataFrame({
        'integer': [1, 2, 1, 3],
        'float': [0.1, 0.2, 0.1, 0.1],
        'categorical': [0.75, 0.25, 0.25, 0.75],
        'bool': [0.0, 0.0, 1.0, 0.0],
        'datetime': [
            1.2649824e+18,
            1.262304e+18,
            1.2649824e+18,
            1.262304e+18
        ],
        'names': [0.25, 0.875, 0.625, 0.25]
    })


def get_transformed_nan_data():
    return pd.DataFrame({
        'integer': [1, 2, 1, 3, 1],
        'float': [0.1, 0.2, 0.1, 0.125, 0.1],
        'float#1': [0.0, 0.0, 0.0, 1.0, 0.0],
        'categorical': [0.6, 0.2, 0.9, 0.2, 0.6],
        'bool': [0.0, -1.0, 0.0, 1.0, 0.0],
        'bool#1': [0.0, 1.0, 0.0, 0.0, 0.0],
        'datetime': [
            1.2636432e+18, 1.2649824e+18, 1.262304e+18,
            1.2649824e+18, 1.262304e+18
        ],
        'datetime#1': [1.0, 0.0, 0.0, 0.0, 0.0],
        'names': [0.2, 0.9, 0.5, 0.2, 0.7],
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


def test_hypertransformer_with_transformers():
    data = get_input_data_without_nan()
    transformers = get_transformers()

    ht = HyperTransformer(transformers)
    ht.fit(data)
    transformed = ht.transform(data)

    expected = get_transformed_data()

    np.testing.assert_allclose(
        transformed.sort_index(axis=1).values,
        expected.sort_index(axis=1).values
    )

    reversed_data = ht.reverse_transform(transformed)

    original_names = data.pop('names')
    reversed_names = reversed_data.pop('names')

    pd.testing.assert_frame_equal(data.sort_index(axis=1), reversed_data.sort_index(axis=1))

    for name in original_names:
        assert name not in reversed_names


def test_hypertransformer_with_transformers_nan_data():
    data = get_input_data_with_nan()
    transformers = get_transformers()

    ht = HyperTransformer(transformers)
    ht.fit(data)
    transformed = ht.transform(data)

    expected = get_transformed_nan_data()

    np.testing.assert_allclose(
        transformed.sort_index(axis=1).values,
        expected.sort_index(axis=1).values
    )

    reversed_data = ht.reverse_transform(transformed)

    original_names = data.pop('names')
    reversed_names = reversed_data.pop('names')

    pd.testing.assert_frame_equal(data.sort_index(axis=1), reversed_data.sort_index(axis=1))

    for name in original_names:
        assert name not in reversed_names


def test_hypertransformer_without_transformers():
    data = get_input_data_without_nan()

    ht = HyperTransformer()
    ht.fit(data)
    transformed = ht.transform(data)

    expected = get_transformed_data()

    np.testing.assert_allclose(
        transformed.sort_index(axis=1).values,
        expected.sort_index(axis=1).values
    )

    reversed_data = ht.reverse_transform(transformed)

    original_names = data.pop('names')
    reversed_names = reversed_data.pop('names')

    pd.testing.assert_frame_equal(data.sort_index(axis=1), reversed_data.sort_index(axis=1))

    for name in original_names:
        assert name not in reversed_names


def test_hypertransformer_without_transformers_nan_data():
    data = get_input_data_with_nan()

    ht = HyperTransformer()
    ht.fit(data)
    transformed = ht.transform(data)

    expected = get_transformed_nan_data()

    np.testing.assert_allclose(
        transformed.sort_index(axis=1).values,
        expected.sort_index(axis=1).values
    )

    reversed_data = ht.reverse_transform(transformed)

    original_names = data.pop('names')
    reversed_names = reversed_data.pop('names')

    pd.testing.assert_frame_equal(data.sort_index(axis=1), reversed_data.sort_index(axis=1))

    for name in original_names:
        assert name not in reversed_names


def test_single_category():
    ht = HyperTransformer(transformers={
        'a': OneHotEncodingTransformer()
    })
    data = pd.DataFrame({
        'a': ['a', 'a', 'a']
    })

    ht.fit(data)
    transformed = ht.transform(data)

    reverse = ht.reverse_transform(transformed)

    pd.testing.assert_frame_equal(data, reverse)


def test_dtype_category():
    df = pd.DataFrame({'a': ['a', 'b', 'c']}, dtype='category')

    ht = HyperTransformer()
    ht.fit(df)

    trans = ht.transform(df)

    rever = ht.reverse_transform(trans)

    pd.testing.assert_frame_equal(df, rever)


def test_empty_transformers():
    """If transformers is an empty dict, do nothing."""
    data = get_input_data_without_nan()

    ht = HyperTransformer(transformers={})
    ht.fit(data)

    transformed = ht.transform(data)
    reverse = ht.reverse_transform(transformed)

    pd.testing.assert_frame_equal(data, transformed)
    pd.testing.assert_frame_equal(data, reverse)


def test_empty_transformers_nan_data():
    """If transformers is an empty dict, do nothing."""
    data = get_input_data_with_nan()

    ht = HyperTransformer(transformers={})
    ht.fit(data)

    transformed = ht.transform(data)
    reverse = ht.reverse_transform(transformed)

    pd.testing.assert_frame_equal(data, transformed)
    pd.testing.assert_frame_equal(data, reverse)


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
