import numpy as np
import pandas as pd

from rdt.transformers import CategoricalTransformer


def test_categorical_numerical_nans():
    """Ensure CategoricalTransformers work on numerical + nan only columns."""

    data = pd.Series([1, 2, float('nan'), np.nan])

    ct = CategoricalTransformer()
    ct.fit(data)
    transformed = ct.transform(data)
    reverse = ct.reverse_transform(transformed)

    pd.testing.assert_series_equal(reverse, data)
