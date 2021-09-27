import pickle
from io import BytesIO
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from rdt.transformers import (
    CategoricalTransformer, LabelEncodingTransformer, OneHotEncodingTransformer)


def test_categorical_numerical_nans():
    """Ensure CategoricalTransformer works on numerical + nan only columns."""

    data = pd.DataFrame([1, 2, float('nan'), np.nan], columns=['column_name'])

    transformer = CategoricalTransformer()
    transformer.fit(data, list(data.columns))
    transformed = transformer.transform(data)
    reverse = transformer.reverse_transform(transformed)

    pd.testing.assert_frame_equal(reverse, data)


def test_categoricaltransformer_pickle_nans():
    """Ensure that CategoricalTransformer can be pickled and loaded with nan value."""
    # setup
    data = pd.DataFrame([1, 2, float('nan'), np.nan], columns=['column_name'])

    transformer = CategoricalTransformer()
    transformer.fit(data, list(data.columns))
    transformed = transformer.transform(data)

    # create pickle file on memory
    bytes_io = BytesIO()
    pickle.dump(transformer, bytes_io)
    # rewind
    bytes_io.seek(0)

    # run
    pickled_transformer = pickle.load(bytes_io)

    # assert
    pickle_transformed = pickled_transformer.transform(data)
    pd.testing.assert_frame_equal(pickle_transformed, transformed)


def test_categoricaltransformer_strings():
    """Test the CategoricalTransformer on string data.

    Ensure that the CategoricalTransformer can fit, transform, and reverse
    transform on string data. Expect that the reverse transformed data
    is the same as the input.

    Input:
        - 4 rows of string data
    Output:
        - The reverse transformed data
    """
    # setup
    data = pd.DataFrame(['a', 'b', 'a', 'c'], columns=['column_name'])
    transformer = CategoricalTransformer()

    # run
    transformer.fit(data, list(data.columns))
    reverse = transformer.reverse_transform(transformer.transform(data))

    # assert
    pd.testing.assert_frame_equal(data, reverse)


def test_categoricaltransformer_strings_2_categories():
    """Test the CategoricalTransformer on string data.

    Ensure that the CategoricalTransformer can fit, transform, and reverse
    transform on string data, when there are 2 categories of strings with
    the same value counts. Expect that the reverse transformed data is the
    same as the input.

    Input:
        - 4 rows of string data
    Output:
        - The reverse transformed data
    """
    # setup
    data = pd.DataFrame(['a', 'b', 'a', 'b'], columns=['column_name'])
    transformer = CategoricalTransformer()

    transformer.fit(data, list(data.columns))
    reverse = transformer.reverse_transform(transformer.transform(data))

    # assert
    pd.testing.assert_frame_equal(data, reverse)


def test_categoricaltransformer_integers():
    """Test the CategoricalTransformer on integer data.

    Ensure that the CategoricalTransformer can fit, transform, and reverse
    transform on integer data. Expect that the reverse transformed data is the
    same as the input.

    Input:
        - 4 rows of int data
    Output:
        - The reverse transformed data
    """
    # setup
    data = pd.DataFrame([1, 2, 3, 2], columns=['column_name'])
    transformer = CategoricalTransformer()

    # run
    transformer.fit(data, list(data.columns))
    reverse = transformer.reverse_transform(transformer.transform(data))

    # assert
    pd.testing.assert_frame_equal(data, reverse)


def test_categoricaltransformer_bool():
    """Test the CategoricalTransformer on boolean data.

    Ensure that the CategoricalTransformer can fit, transform, and reverse
    transform on boolean data. Expect that the reverse transformed data is the
    same as the input.

    Input:
        - 4 rows of bool data
    Output:
        - The reverse transformed data
    """
    # setup
    data = pd.DataFrame([True, False, True, False], columns=['column_name'])
    transformer = CategoricalTransformer()

    # run
    transformer.fit(data, list(data.columns))
    reverse = transformer.reverse_transform(transformer.transform(data))

    # assert
    pd.testing.assert_frame_equal(data, reverse)


def test_categoricaltransformer_mixed():
    """Test the CategoricalTransformer on mixed type data.

    Ensure that the CategoricalTransformer can fit, transform, and reverse
    transform on mixed type data. Expect that the reverse transformed data is
    the same as the input.

    Input:
        - 4 rows of mixed data
    Output:
        - The reverse transformed data
    """
    # setup
    data = pd.DataFrame([True, 'a', 1, None], columns=['column_name'])
    transformer = CategoricalTransformer()

    # run
    transformer.fit(data, list(data.columns))
    reverse = transformer.reverse_transform(transformer.transform(data))

    # assert
    pd.testing.assert_frame_equal(data, reverse)


@patch('psutil.virtual_memory')
def test_categoricaltransformer_mixed_low_virtual_memory(psutil_mock):
    """Test the CategoricalTransformer on mixed type data with low virtual memory.

    Ensure that the CategoricalTransformer can fit, transform, and reverse
    transform on mixed type data, when there is low virtual memory. Expect that the
    reverse transformed data is the same as the input.

    Input:
        - 4 rows of mixed data
    Output:
        - The reverse transformed data
    """
    # setup
    data = pd.DataFrame([True, 'a', 1, None], columns=['column_name'])
    transformer = CategoricalTransformer()

    virtual_memory = Mock()
    virtual_memory.available = 1
    psutil_mock.return_value = virtual_memory

    # run
    transformer.fit(data, list(data.columns))
    reverse = transformer.reverse_transform(transformer.transform(data))

    # assert
    pd.testing.assert_frame_equal(data, reverse)


@patch('psutil.virtual_memory')
def test_categoricaltransformer_mixed_more_rows(psutil_mock):
    """Test the CategoricalTransformer on mixed type data with low virtual memory.

    Ensure that the CategoricalTransformer can fit, transform, and reverse
    transform on mixed type data, when there is low virtual memory and a larger
    number of rows. Expect that the reverse transformed data is the same as the input.

    Input:
        - 4 rows of mixed data
    Output:
        - The reverse transformed data
    """
    # setup
    data = pd.DataFrame([True, 'a', 1, None], columns=['column_name'])
    transform_data = pd.DataFrame(['a', 1, None, 'a', True, 1], columns=['column_name'])
    transformer = CategoricalTransformer()

    virtual_memory = Mock()
    virtual_memory.available = 1
    psutil_mock.return_value = virtual_memory

    # run
    transformer.fit(data, list(data.columns))
    transformed = transformer.transform(transform_data)
    reverse = transformer.reverse_transform(transformed)

    # assert
    pd.testing.assert_frame_equal(transform_data, reverse)


def test_one_hot_numerical_nans():
    """Ensure OneHotEncodingTransformer works on numerical + nan only columns."""

    data = pd.DataFrame([1, 2, float('nan'), np.nan], columns=['column_name'])

    transformer = OneHotEncodingTransformer()
    transformer.fit(data, list(data.columns))
    transformed = transformer.transform(data)
    reverse = transformer.reverse_transform(transformed)

    pd.testing.assert_frame_equal(reverse, data)


def test_label_numerical_2d_array():
    """Ensure LabelEncodingTransformer works on numerical + nan only columns."""

    data = pd.DataFrame(['a', 'b', 'c', 'd'], columns=['column_name'])

    transformer = LabelEncodingTransformer()
    transformer.fit(data, list(data.columns))
    transformed = pd.DataFrame([0, 1, 2, 3], columns=['column_name.value'])
    reverse = transformer.reverse_transform(transformed)

    pd.testing.assert_frame_equal(reverse, data)


def test_label_numerical_nans():
    """Ensure LabelEncodingTransformer works on numerical + nan only columns."""

    data = pd.DataFrame([1, 2, float('nan'), np.nan], columns=['column_name'])

    transformer = LabelEncodingTransformer()
    transformer.fit(data, list(data.columns))
    transformed = transformer.transform(data)
    reverse = transformer.reverse_transform(transformed)

    pd.testing.assert_frame_equal(reverse, data)
