import numpy as np
import pandas as pd

from rdt.transformers.datetime import DatetimeRoundedTransformer, DatetimeTransformer


class TestDatetimeTransformer:

    def test__reverse_transform_all_none(self):
        dt = pd.to_datetime(['2020-01-01'])
        dtt = DatetimeTransformer(strip_constant=True)
        dtt._fit(dt)

        output = dtt._reverse_transform(np.array([None]))

        expected = pd.to_datetime(['NaT'])
        pd.testing.assert_series_equal(output.to_series(), expected.to_series())

    def test__reverse_transform_2d_ndarray(self):
        dt = pd.to_datetime(['2020-01-01', '2020-02-01', '2020-03-01'])
        dtt = DatetimeTransformer(strip_constant=True)
        dtt._fit(dt)

        transformed = np.array([[18262.], [18293.], [18322.]])
        output = dtt._reverse_transform(transformed)

        expected = pd.to_datetime(['2020-01-01', '2020-02-01', '2020-03-01'])
        pd.testing.assert_series_equal(output.to_series(), expected.to_series())


class TestDatetimeRoundedTransformer:

    def test___init___strip_is_true(self):
        """Test that by default the ``strip_constant`` is set to True."""
        dtrt = DatetimeRoundedTransformer()

        # assert
        assert dtrt.strip_constant
