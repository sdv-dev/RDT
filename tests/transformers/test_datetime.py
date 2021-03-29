import numpy as np
import pandas as pd

from rdt.transformers import DatetimeTransformer


class TestDatetimeTransformer:

    def test_no_strip(self):
        dtt = DatetimeTransformer(strip_constant=False)
        data = pd.to_datetime(pd.Series([None, '1996-10-17', '1965-05-23']))

        # Run
        transformed = dtt.fit_transform(data.copy().to_numpy())
        reverted = dtt.reverse_transform(transformed)

        # Asserts
        expect_trans = np.array([
            [350006400000000000, 1.0],
            [845510400000000000, 0.0],
            [-145497600000000000, 0.0]
        ])
        np.testing.assert_almost_equal(expect_trans, transformed)
        pd.testing.assert_series_equal(reverted, data)

    def test_strip(self):
        dtt = DatetimeTransformer(strip_constant=True)
        data = pd.to_datetime(pd.Series([None, '1996-10-17', '1965-05-23']))

        # Run
        transformed = dtt.fit_transform(data.copy().to_numpy())
        reverted = dtt.reverse_transform(transformed)

        # Asserts
        expect_trans = np.array([
            [4051.0, 1.0],
            [9786.0, 0.0],
            [-1684.0, 0.0]
        ])
        np.testing.assert_almost_equal(expect_trans, transformed)
        pd.testing.assert_series_equal(reverted, data)

    def test_reverse_transform_all_none(self):
        dt = pd.to_datetime(['2020-01-01'])
        dtt = DatetimeTransformer(strip_constant=True)
        dtt.fit(dt)

        output = dtt.reverse_transform(np.array([None]))

        expected = pd.to_datetime(['NaT'])
        pd.testing.assert_series_equal(output.to_series(), expected.to_series())

    def test_reverse_transform_2d_ndarray(self):
        dt = pd.to_datetime(['2020-01-01', '2020-02-01', '2020-03-01'])
        dtt = DatetimeTransformer(strip_constant=True)
        dtt.fit(dt)

        transformed = np.array([[18262.], [18293.], [18322.]])
        output = dtt.reverse_transform(transformed)

        expected = pd.to_datetime(['2020-01-01', '2020-02-01', '2020-03-01'])
        pd.testing.assert_series_equal(output.to_series(), expected.to_series())
