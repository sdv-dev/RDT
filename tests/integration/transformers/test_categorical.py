import pickle
from io import BytesIO

import numpy as np
import pandas as pd

from rdt.transformers import (
    CategoricalTransformer, LabelEncodingTransformer, OneHotEncodingTransformer)


def test_categorical_numerical_nans():
    """Ensure CategoricalTransformer works on numerical + nan only columns."""

    data = pd.Series([1, 2, float('nan'), np.nan])

    transformer = CategoricalTransformer()
    transformer.fit(data)
    transformed = transformer.transform(data)
    reverse = transformer.reverse_transform(transformed)

    pd.testing.assert_series_equal(reverse, data)


def test_categoricaltransformer_pickle_nans():
    """Ensure that CategoricalTransformer can be pickled and loaded with nan value."""
    # setup
    data = pd.Series([1, 2, float('nan'), np.nan])

    transformer = CategoricalTransformer()
    transformer.fit(data)
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
    np.testing.assert_array_equal(pickle_transformed, transformed)


def test_one_hot_numerical_nans():
    """Ensure OneHotEncodingTransformer works on numerical + nan only columns."""

    data = pd.Series([1, 2, float('nan'), np.nan])

    transformer = OneHotEncodingTransformer()
    transformer.fit(data)
    transformed = transformer.transform(data)
    reverse = transformer.reverse_transform(transformed)

    pd.testing.assert_series_equal(reverse, data)


def test_label_numerical_2d_array():
    """Ensure LabelEncodingTransformer works on numerical + nan only columns."""

    data = pd.Series([1, 2, 3, 4])

    transformer = LabelEncodingTransformer()
    transformer.fit(data)
    transformed = np.array([[0], [1], [2], [3]])
    reverse = transformer.reverse_transform(transformed)

    pd.testing.assert_series_equal(reverse, data)


def test_label_numerical_nans():
    """Ensure LabelEncodingTransformer works on numerical + nan only columns."""

    data = pd.Series([1, 2, float('nan'), np.nan])

    transformer = LabelEncodingTransformer()
    transformer.fit(data)
    transformed = transformer.transform(data)
    reverse = transformer.reverse_transform(transformed)

    pd.testing.assert_series_equal(reverse, data)
