import numpy as np
import pandas as pd

from rdt.transformers.datetime import DatetimeTransformer


def test_no_strip():
    dtt = DatetimeTransformer(strip_constant=False)
    data = pd.to_datetime(pd.Series([None, '1996-10-17', '1965-05-23']))

    # Run
    dtt._fit(data.copy())
    transformed = dtt._transform(data.copy())
    reverted = dtt._reverse_transform(transformed)

    # Asserts
    expect_trans = np.array([
        [350006400000000000, 1.0],
        [845510400000000000, 0.0],
        [-145497600000000000, 0.0]
    ])
    np.testing.assert_almost_equal(expect_trans, transformed)
    pd.testing.assert_series_equal(reverted, data)


def test_strip():
    dtt = DatetimeTransformer(strip_constant=True)
    data = pd.to_datetime(pd.Series([None, '1996-10-17', '1965-05-23']))

    # Run
    dtt._fit(data.copy())
    transformed = dtt._transform(data.copy())
    reverted = dtt._reverse_transform(transformed)

    # Asserts
    expect_trans = np.array([
        [4051.0, 1.0],
        [9786.0, 0.0],
        [-1684.0, 0.0]
    ])
    np.testing.assert_almost_equal(expect_trans, transformed)
    pd.testing.assert_series_equal(reverted, data)
