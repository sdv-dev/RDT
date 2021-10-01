import numpy as np
import pandas as pd
import pytest

from rdt import HyperTransformer
from rdt.transformers import OneHotEncodingTransformer


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


@pytest.mark.xfail
def test_dtype_category():
    df = pd.DataFrame({'a': ['a', 'b', 'c']}, dtype='category')

    ht = HyperTransformer()
    ht.fit(df)

    trans = ht.transform(df)

    rever = ht.reverse_transform(trans)

    pd.testing.assert_frame_equal(rever, df)


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


#----------------#----------------#----------------#----------------#----------------#----------------


def null_transformer_asserts(data, ht, transformed, expected):
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

def test_hypertransformer_transform_nulls_false():
    data = get_input_data_without_nan()
    transformers = get_transformers()

    ht = HyperTransformer(transformers, transform_nulls=False)
    ht.fit(data)
    transformed = ht.transform(data)

    expected = get_transformed_data()
    null_transformer_asserts(data, ht, transformed, expected)


def test_hypertransformer_fill_value_None():
    data = get_input_data_without_nan()
    transformers = get_transformers()

    ht = HyperTransformer(transformers, fill_value=np.nan)
    raise TypeError('fill_value cant be np.nan')
    

def get_transformed_fill_value_string():
    return pd.DataFrame({
        'integer': [1, 2, 1, 3, 1],
        'float': [0.1, 0.2, 0.1, 'filled_value', 0.1],
        'categorical': [0.3, 0.3, 'filled_value', 0.7, 0.3],
        'bool': [0.0, 'filled_value', 0.0, 1.0, 0.0],
        'datetime': [
            'filled_value', 1.2649824e+18, 1.262304e+18,
            1.2649824e+18, 1.262304e+18
        ],
        'names': [0.3, 0.8, 0.8, 0.3, 0.3],
    })


def test_hypertransformer_fill_value_string():
    data = get_input_data_with_nan()
    transformers = get_transformers()

    ht = HyperTransformer(transformers, fill_value='filled_value')
    ht.fit(data)
    transformed = ht.transform(data)

    expected = get_transformed_fill_value_string()
    null_transformer_asserts(data, ht, transformed, expected)


def get_transformed_fill_value_mean():
    """TODO: Change the 'mean' values with the actual mean."""
    return pd.DataFrame({
        'integer': [1, 2, 1, 3, 1],
        'float': [0.1, 0.2, 0.1, 'mean', 0.1],
        'categorical': [0.3, 0.3, 'mean', 0.7, 0.3],
        'bool': [0.0, 'mean', 0.0, 1.0, 0.0],
        'datetime': [
            'mean', 1.2649824e+18, 1.262304e+18,
            1.2649824e+18, 1.262304e+18
        ],
        'names': [0.3, 0.8, 0.8, 0.3, 0.3],
    })

def test_hypertransformer_fill_value_mean():
    data = get_input_data_with_nan()
    transformers = get_transformers()

    ht = HyperTransformer(transformers, fill_value='mean')
    ht.fit(data)
    transformed = ht.transform(data)

    expected = get_transformed_fill_value_mean()
    null_transformer_asserts(data, ht, transformed, expected)


def get_transformed_fill_value_mode():
    """TODO: Change the 'mode' values with the actual mean."""
    return pd.DataFrame({
        'integer': [1, 2, 1, 3, 1],
        'float': [0.1, 0.2, 0.1, 'mode', 0.1],
        'categorical': [0.3, 0.3, 'mode', 0.7, 0.3],
        'bool': [0.0, 'mode', 0.0, 1.0, 0.0],
        'datetime': [
            'mode', 1.2649824e+18, 1.262304e+18,
            1.2649824e+18, 1.262304e+18
        ],
        'names': [0.3, 0.8, 0.8, 0.3, 0.3],
    })

def test_hypertransformer_fill_value_mode():
    data = get_input_data_with_nan()
    transformers = get_transformers()

    ht = HyperTransformer(transformers, fill_value='mode')
    ht.fit(data)
    transformed = ht.transform(data)

    expected = get_transformed_fill_value_mode()
    null_transformer_asserts(data, ht, transformed, expected)


def get_transformed_fill_value_dict():
    return pd.DataFrame({
        'integer': [1, 2, 1, 3, 1],
        'float': [0.1, 0.2, 0.1, 0.1, 0.1],
        'categorical': [0.3, 0.3, 0.4, 0.7, 0.3],
        'bool': [0.0, 'filled_value', 0.0, 1.0, 0.0],
        'datetime': [
            np.nan, 1.2649824e+18, 1.262304e+18,
            1.2649824e+18, 1.262304e+18
        ],
        'names': [0.3, 0.8, 0.8, 0.3, 0.3],
    })

def test_hypertransformer_fill_value_mode():
    data = get_input_data_with_nan()
    transformers = get_transformers()

    ht = HyperTransformer(transformers, fill_value={'float': 'mode', 'categorical': 'mean', 'bool': 'filled_value'})
    ht.fit(data)
    transformed = ht.transform(data)

    expected = get_transformed_fill_value_dict()
    null_transformer_asserts(data, ht, transformed, expected)


def test_hypertransformer_fill_value_dict_wrong_column():
    data = get_input_data_with_nan()
    transformers = get_transformers()

    ht = HyperTransformer(transformers, fill_value={'non_existing_column': 'mean'})
    ht.fit(data)
    raise ValueError

#----------------#----------------#----------------#----------------#----------------#----------------

def get_transformed_fill_value_string_null_column_True():
    return pd.DataFrame({
        'integer': [1, 2, 1, 3, 1],
        'float': [0.1, 0.2, 0.1, 'filled_value', 0.1],
        'float.null': [0.0, 0.0, 0.0, 1.0, 0.0],
        'categorical': [0.3, 0.3, 'filled_value', 0.7, 0.3],
        'categorical.null': [0.0, 0.0, 1.0, 0.0, 0.0],
        'bool': [0.0, 'filled_value', 0.0, 1.0, 0.0],
        'bool.null': [0.0, 1.0, 0.0, 0.0, 0.0],
        'datetime': [
            'filled_value', 1.2649824e+18, 1.262304e+18,
            1.2649824e+18, 1.262304e+18
        ],
        'datetime.null': [1.0, 0.0, 0.0, 0.0, 0.0],
        'names': [0.3, 0.8, 0.8, 0.3, 0.3],
    })

def test_hypertransformer_fill_value_string_null_column_True():
    data = get_input_data_with_nan()
    transformers = get_transformers()

    ht = HyperTransformer(transformers, fill_value='filled_value', null_column=True)
    ht.fit(data)
    transformed = ht.transform(data)

    expected = get_transformed_fill_value_string_null_column_True()
    null_transformer_asserts(data, ht, transformed, expected)


def get_transformed_fill_value_dict_null_column_True():
    return pd.DataFrame({
        'integer': [1, 2, 1, 3, 1],
        'float': [0.1, 0.2, 0.1, 0.1, 0.1],
        'float.null': [0.0, 0.0, 0.0, 1.0, 0.0],       
        'categorical': [0.3, 0.3, 0.4, 0.7, 0.3],
        'categorical.null': [0.0, 0.0, 1.0, 0.0, 0.0],
        'bool': [0.0, 'filled_value', 0.0, 1.0, 0.0],
        'bool.null': [0.0, 1.0, 0.0, 0.0, 0.0],
        'datetime': [
            np.nan, 1.2649824e+18, 1.262304e+18,
            1.2649824e+18, 1.262304e+18
        ],
        'datetime.null': [1.0, 0.0, 0.0, 0.0, 0.0],
        'names': [0.3, 0.8, 0.8, 0.3, 0.3],
    })

def test_hypertransformer_fill_value_dict_mode_null_column_True():
    data = get_input_data_with_nan()
    transformers = get_transformers()

    ht = HyperTransformer(transformers, fill_value={'float': 'mode', 'categorical': 'mean', 'bool': 'filled_value'})
    ht.fit(data)
    transformed = ht.transform(data)

    expected = get_transformed_fill_value_dict_null_column_True()
    null_transformer_asserts(data, ht, transformed, expected)


#----------------#----------------#----------------#----------------#----------------#----------------


def get_transformed_fill_value_string_null_column_dict():
    return pd.DataFrame({
        'integer': [1, 2, 1, 3, 1],
        'float': [0.1, 0.2, 0.1, 'filled_value', 0.1],
        'float.null': [0.0, 0.0, 0.0, 1.0, 0.0],
        'categorical': [0.3, 0.3, 'filled_value', 0.7, 0.3],
        'categorical.null': [0.0, 0.0, 1.0, 0.0, 0.0],
        'bool': [0.0, 'filled_value', 0.0, 1.0, 0.0],
        'datetime': [
            'filled_value', 1.2649824e+18, 1.262304e+18,
            1.2649824e+18, 1.262304e+18
        ],
        'names': [0.3, 0.8, 0.8, 0.3, 0.3],
    })

def test_hypertransformer_fill_value_string_null_column_True():
    data = get_input_data_with_nan()
    transformers = get_transformers()

    ht = HyperTransformer(transformers, fill_value='filled_value', null_column={'float': True, 'categorical': True})
    ht.fit(data)
    transformed = ht.transform(data)

    expected = get_transformed_fill_value_string_null_column_True()
    null_transformer_asserts(data, ht, transformed, expected)


def get_transformed_fill_value_dict_null_column_True():
    return pd.DataFrame({
        'integer': [1, 2, 1, 3, 1],
        'float': [0.1, 0.2, 0.1, 0.1, 0.1],
        'categorical': [0.3, 0.3, 0.4, 0.7, 0.3],
        'categorical.null': [0.0, 0.0, 1.0, 0.0, 0.0],
        'bool': [0.0, 'filled_value', 0.0, 1.0, 0.0],
        'datetime': [
            np.nan, 1.2649824e+18, 1.262304e+18,
            1.2649824e+18, 1.262304e+18
        ],
        'datetime.null': [1.0, 0.0, 0.0, 0.0, 0.0],
        'names': [0.3, 0.8, 0.8, 0.3, 0.3],
    })

def test_hypertransformer_fill_value_dict_mode_null_column_True():
    data = get_input_data_with_nan()
    transformers = get_transformers()

    ht = HyperTransformer(transformers, fill_value={'float': 'mode', 'categorical': 'mean', 'bool': 'filled_value'},
                          null_column={'float': False, 'categorical': True, 'datetime': True})
    ht.fit(data)
    transformed = ht.transform(data)

    expected = get_transformed_fill_value_dict_null_column_True()
    null_transformer_asserts(data, ht, transformed, expected)