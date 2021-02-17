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


def test_one_hot_numerical_nans():
    """Ensure OneHotEncodingTransformer works on numerical + nan only columns."""

    data = pd.Series([1, 2, float('nan'), np.nan])

    transformer = OneHotEncodingTransformer()
    transformer.fit(data)
    transformed = transformer.transform(data)
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
