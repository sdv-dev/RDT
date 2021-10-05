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
    """Test the HyperTransformer with ``transform_nulls = False``.

    When ``transform_nulls`` is ``False``, should not applying the ``NullTransformer`` at all.

    Setup:
        - Get the data and the transformers.

    Input:
        - A dataset without ``nan`` values.
        - A dictionary of which transformers to apply to each column of the data.

    Expected behavior:
        - It should fit and transform the dataset.
        - The results will be checked through the ``null_transformer_asserts`` method.
    """
    data = get_input_data_without_nan()
    transformers = get_transformers()

    ht = HyperTransformer(transformers, transform_nulls=False)
    ht.fit(data)
    transformed = ht.transform(data)

    expected = get_transformed_data()
    null_transformer_asserts(data, ht, transformed, expected)


def test_hypertransformer_transform_nulls_false_fill_value_mean():
    """Test the HyperTransformer with ``transform_nulls = False`` and ``fill_value = 'mean'``.

    When ``transform_nulls = False``, if ``fill_value`` is passed as anything other than None,
    it should raise an error.

    Setup:
        - Get the data and the transformers.

    Input:
        - A dataset without ``nan`` values.
        - A dictionary of which transformers to apply to each column of the data.

    Expected behavior:
        - It should fit and transform the dataset.
        - The results will be checked through the ``null_transformer_asserts`` method.
    """
    data = get_input_data_without_nan()
    transformers = get_transformers()
    
    with np.testing.assert_raises(ValueError):
        HyperTransformer(transformers, transform_nulls=False, fill_value='mean')


def test_hypertransformer_transform_nulls_false_null_column_true():
    """Test the HyperTransformer with ``transform_nulls = False`` and ``null_column = 'mean'``.

    When ``transform_nulls = False``, if ``null_column`` is passed as anything other than None,
    it should raise an error.

    Setup:
        - Get the data and the transformers.

    Input:
        - A dataset without ``nan`` values.
        - A dictionary of which transformers to apply to each column of the data.

    Expected behavior:
        - It should fit and transform the dataset.
        - The results will be checked through the ``null_transformer_asserts`` method.
    """
    data = get_input_data_without_nan()
    transformers = get_transformers()
    
    with np.testing.assert_raises(ValueError):
        HyperTransformer(transformers, transform_nulls=False, null_column=True)


# TODO: what should be the behaviour here? What if we pass False as fill_value?
def test_hypertransformer_fill_value_nan():
    """Test the HyperTransformer with ``fill_value = np.nan``.

    ???

    Setup:
        - Get the data and the transformers.

    Input:
        - A dataset without ``nan`` values.
        - A dictionary of which transformers to apply to each column of the data.

    Expected behavior:
        - It should fit and transform the dataset.
        - The results will be checked through the ``null_transformer_asserts`` method.
    """
    data = get_input_data_without_nan()
    transformers = get_transformers()

    with np.testing.assert_raises(ValueError):
        HyperTransformer(transformers, fill_value=np.nan)
    

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
    """Test the HyperTransformer with ``fill_value = string``.

    When ``fill_value`` is a string ``'filled_value'``, it should behave like the normal
    ``HyperTransformer``, but filling all the transformed ``np.nan`` values with the
    string ``'filled_value'``.

    Setup:
        - Get the data and the transformers.

    Input:
        - A dataset without ``nan`` values.
        - A dictionary of which transformers to apply to each column of the data.

    Expected behavior:
        - It should fit and transform the dataset.
        - The results will be checked through the ``null_transformer_asserts`` method.
    """
    data = get_input_data_with_nan()
    transformers = get_transformers()

    ht = HyperTransformer(transformers, fill_value='filled_value')
    ht.fit(data)
    transformed = ht.transform(data)

    expected = get_transformed_fill_value_string()
    null_transformer_asserts(data, ht, transformed, expected)


def get_transformed_fill_value_mean():
    return pd.DataFrame({
        'integer': [1, 2, 1, 3, 1],
        'float': [0.1, 0.2, 0.1, 0.125, 0.1],
        'categorical': [0.3, 0.3, 0.4, 0.7, 0.3],
        'bool': [0.0, 0.25, 0.0, 1.0, 0.0],
        'datetime': [
            1.2636432e+18, 1.2649824e+18, 1.262304e+18,
            1.2649824e+18, 1.262304e+18
        ],
        'names': [0.3, 0.8, 0.8, 0.3, 0.3],
    })


def test_hypertransformer_fill_value_mean():
    """Test the HyperTransformer with ``fill_value = mean``.

    When ``fill_value`` is the string ``'mean'``, it should behave like the normal
    ``HyperTransformer``, but filling all the transformed ``np.nan`` values with the
    mean of the values of the column.

    Setup:
        - Get the data and the transformers.

    Input:
        - A dataset without ``nan`` values.
        - A dictionary of which transformers to apply to each column of the data.

    Expected behavior:
        - It should fit and transform the dataset.
        - The results will be checked through the ``null_transformer_asserts`` method.
    """
    data = get_input_data_with_nan()
    transformers = get_transformers()

    ht = HyperTransformer(transformers, fill_value='mean')
    ht.fit(data)
    transformed = ht.transform(data)

    expected = get_transformed_fill_value_mean()
    null_transformer_asserts(data, ht, transformed, expected)


def get_transformed_fill_value_mode():
    return pd.DataFrame({
        'integer': [1, 2, 1, 3, 1],
        'float': [0.1, 0.2, 0.1, 0.1, 0.1],
        'categorical': [0.3, 0.3, 0.3, 0.7, 0.3],
        'bool': [0.0, 0.0, 0.0, 1.0, 0.0],
        'datetime': [
            1.2649824e+18, 1.2649824e+18, 1.262304e+18,  # The first value is the mode (I hope?)
            1.2649824e+18, 1.262304e+18
        ],
        'names': [0.3, 0.8, 0.8, 0.3, 0.3],
    })


def test_hypertransformer_fill_value_mode():
    """Test the HyperTransformer with ``fill_value = mode``.

    When ``fill_value`` is the string ``'mode'``, it should behave like the normal
    ``HyperTransformer``, but filling all the transformed ``np.nan`` values with the
    mode of the values of the column.

    Setup:
        - Get the data and the transformers.

    Input:
        - A dataset without ``nan`` values.
        - A dictionary of which transformers to apply to each column of the data.

    Expected behavior:
        - It should fit and transform the dataset.
        - The results will be checked through the ``null_transformer_asserts`` method.
    """
    data = get_input_data_with_nan()
    transformers = get_transformers()

    ht = HyperTransformer(transformers, fill_value='mode')
    ht.fit(data)
    transformed = ht.transform(data)

    expected = get_transformed_fill_value_mode()
    null_transformer_asserts(data, ht, transformed, expected)


def get_transformed_fill_value_object():
    return pd.DataFrame({
        'integer': [1, 2, 1, 3, 1],
        'float': [0.1, 0.2, 0.1, {'key': 'value'}, 0.1],
        'categorical': [0.3, 0.3, {'key': 'value'}, 0.7, 0.3],
        'bool': [0.0, {'key': 'value'}, 0.0, 1.0, 0.0],
        'datetime': [
            {'key': 'value'}, 1.2649824e+18, 1.262304e+18,
            1.2649824e+18, 1.262304e+18
        ],
        'names': [0.3, 0.8, 0.8, 0.3, 0.3],
    })


def test_hypertransformer_fill_value_object():
    """Test the HyperTransformer with ``fill_value = object``.

    When ``fill_value`` is an object, like ``{'key': 'value'}``, it should behave like the
    normal ``HyperTransformer``, but filling all the transformed ``np.nan`` values with the
    ``{'key': 'value'}`` object.

    Setup:
        - Get the data and the transformers.

    Input:
        - A dataset without ``nan`` values.
        - A dictionary of which transformers to apply to each column of the data.

    Expected behavior:
        - It should fit and transform the dataset.
        - The results will be checked through the ``null_transformer_asserts`` method.
    """
    data = get_input_data_with_nan()
    transformers = get_transformers()

    ht = HyperTransformer(transformers, fill_value={'key': 'value'})
    ht.fit(data)
    transformed = ht.transform(data)

    expected = get_transformed_fill_value_object()
    null_transformer_asserts(data, ht, transformed, expected)


def get_transformed_fill_value_string_null_column_True():
    return pd.DataFrame({
        'integer': [1, 2, 1, 3, 1],
        'integer.null': [0.0, 0.0, 0.0, 0.0, 0.0],
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
        'names.null': [0.0, 0.0, 0.0, 0.0, 0.0]
    })


def test_hypertransformer_fill_value_string_null_column_True():
    """Test the HyperTransformer with ``fill_value = string, null_column = True``.

    When ``fill_value`` is a string ``'filled_value'`` and ``null_column`` is ``True``,
    it should (1) transform the data like the normal ``HyperTransformer``, (2) fill all the
    transformed ``np.nan`` values with the ``'filled_value'`` string and (3) create a new
    column for each column in the data, flagging which values contain ``nan``'s .

    Setup:
        - Get the data and the transformers.

    Input:
        - A dataset without ``nan`` values.
        - A dictionary of which transformers to apply to each column of the data.

    Expected behavior:
        - It should fit and transform the dataset.
        - The results will be checked through the ``null_transformer_asserts`` method.
    """
    data = get_input_data_with_nan()
    transformers = get_transformers()

    ht = HyperTransformer(transformers, fill_value='filled_value', null_column=True)
    ht.fit(data)
    transformed = ht.transform(data)

    expected = get_transformed_fill_value_string_null_column_True()
    null_transformer_asserts(data, ht, transformed, expected)


def get_transformed_fill_value_none_null_column_true():
    return pd.DataFrame({
        'integer': [1, 2, 1, 3, 1],
        'integer.null': [0.0, 0.0, 0.0, 0.0, 0.0],
        'float': [0.1, 0.2, 0.1, np.nan, 0.1],
        'float.null': [0.0, 0.0, 0.0, 1.0, 0.0],       
        'categorical': [0.3, 0.3, np.nan, 0.7, 0.3],
        'categorical.null': [0.0, 0.0, 1.0, 0.0, 0.0],
        'bool': [0.0, np.nan, 0.0, 1.0, 0.0],
        'bool.null': [0.0, 1.0, 0.0, 0.0, 0.0],
        'datetime': [
            np.nan, 1.2649824e+18, 1.262304e+18,
            1.2649824e+18, 1.262304e+18
        ],
        'datetime.null': [1.0, 0.0, 0.0, 0.0, 0.0],
        'names': [0.3, 0.8, 0.8, 0.3, 0.3],
        'names.null': [0.0, 0.0, 0.0, 0.0, 0.0]
    })


def test_hypertransformer_fill_value_none_null_column_true():
    """Test the HyperTransformer with ``transform_nulls = False``.

    When ``transform_nulls`` is ``False``, should not applying the ``NullTransformer`` at all.

    Setup:
        - Get the data and the transformers.

    Input:
        - A dataset without ``nan`` values.
        - A dictionary of which transformers to apply to each column of the data.

    Expected behavior:
        - It should fit and transform the dataset.
        - The results will be checked through the ``null_transformer_asserts`` method.
    """
    data = get_input_data_with_nan()
    transformers = get_transformers()

    ht = HyperTransformer(transformers, fill_value=None, null_column=True)
    ht.fit(data)
    transformed = ht.transform(data)

    expected = get_transformed_fill_value_none_null_column_true()
    null_transformer_asserts(data, ht, transformed, expected)


def get_transformed_fill_value_string_null_column_none():
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
        'names': [0.3, 0.8, 0.8, 0.3, 0.3]
    })


def test_hypertransformer_fill_value_string_null_column_none():
    """Test the HyperTransformer with ``transform_nulls = False``.

    When ``transform_nulls`` is ``False``, should not applying the ``NullTransformer`` at all.

    Setup:
        - Get the data and the transformers.

    Input:
        - A dataset without ``nan`` values.
        - A dictionary of which transformers to apply to each column of the data.

    Expected behavior:
        - It should fit and transform the dataset.
        - The results will be checked through the ``null_transformer_asserts`` method.
    """
    data = get_input_data_with_nan()
    transformers = get_transformers()

    ht = HyperTransformer(transformers, fill_value='filled_value', null_column=None)
    ht.fit(data)
    transformed = ht.transform(data)

    expected = get_transformed_fill_value_string_null_column_True()
    null_transformer_asserts(data, ht, transformed, expected)
