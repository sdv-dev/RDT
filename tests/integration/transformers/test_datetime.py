import numpy as np
import pandas as pd

from rdt.transformers.datetime import DatetimeTransformer


def test_no_strip():
    dtt = DatetimeTransformer(missing_value_replacement='mean', strip_constant=False)
    data = pd.to_datetime(pd.Series([None, '1996-10-17', '1965-05-23']))
    dtt.columns = 'column'

    # Run
    dtt._fit(data.copy())
    transformed = dtt._transform(data.copy())
    reverted = dtt._reverse_transform(transformed)

    # Asserts
    expect_trans = np.array([
        np.nan,
        845510400000000000,
        -145497600000000000
    ])
    np.testing.assert_almost_equal(expect_trans, transformed)
    pd.testing.assert_series_equal(reverted, data)


def test_strip():
    dtt = DatetimeTransformer(missing_value_replacement='mean', strip_constant=True)
    data = pd.to_datetime(pd.Series([None, '1996-10-17', '1965-05-23']))
    dtt.columns = 'column'

    # Run
    dtt._fit(data.copy())
    transformed = dtt._transform(data.copy())
    reverted = dtt._reverse_transform(transformed)

    # Asserts
    expect_trans = np.array([
        np.nan,
        9786.0,
        -1684.0
    ])
    np.testing.assert_almost_equal(expect_trans, transformed)
    pd.testing.assert_series_equal(reverted, data)
