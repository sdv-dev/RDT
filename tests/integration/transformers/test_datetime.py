import numpy as np
import pandas as pd

from rdt.transformers.datetime import OptimizedTimestampEncoder, UnixTimestampEncoder


def test_unixtimestampencoder():
    np.random.seed(7)
    ute = UnixTimestampEncoder(missing_value_replacement='mean')
    data = pd.to_datetime(pd.Series([None, '1996-10-17', '1965-05-23']))

    # Run
    ute._fit(data.copy())
    transformed = ute._transform(data.copy())
    reverted = ute._reverse_transform(transformed.copy())

    # Asserts
    expect_transformed = np.array([
        3.500064e+17,
        845510400000000000,
        -145497600000000000
    ])
    np.testing.assert_almost_equal(expect_transformed, transformed)
    pd.testing.assert_series_equal(reverted, data)


def test_unixtimestampencoder_different_format():
    np.random.seed(7)
    ute = UnixTimestampEncoder(missing_value_replacement='mean', datetime_format='%b %d, %Y')
    data = pd.Series([None, 'Oct 17, 1996', 'May 23, 1965'])

    # Run
    ute._fit(data.copy())
    transformed = ute._transform(data.copy())
    reverted = ute._reverse_transform(transformed.copy())

    # Asserts
    expect_trans = np.array([
        3.500064e+17,
        845510400000000000,
        -145497600000000000
    ])
    np.testing.assert_almost_equal(expect_trans, transformed)
    pd.testing.assert_series_equal(reverted, data)


def test_optimizedtimestampencoder():
    np.random.seed(7)
    ote = OptimizedTimestampEncoder(missing_value_replacement='mean')
    data = pd.to_datetime(pd.Series([None, '1996-10-17', '1965-05-23']))
    ote.columns = 'column'

    # Run
    ote._fit(data.copy())
    transformed = ote._transform(data.copy())
    reverted = ote._reverse_transform(transformed.copy())

    # Asserts
    expect_trans = np.array([
        4051.0,
        9786.0,
        -1684.0
    ])
    np.testing.assert_almost_equal(expect_trans, transformed)
    pd.testing.assert_series_equal(reverted, data)
