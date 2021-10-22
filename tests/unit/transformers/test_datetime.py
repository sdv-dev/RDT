import numpy as np
import pandas as pd

from rdt.transformers.datetime import DatetimeRoundedTransformer, DatetimeTransformer


class TestDatetimeTransformer:

    def test__reverse_transform_all_none(self):
        dt = pd.to_datetime(['2020-01-01'])
        dtt = DatetimeTransformer(strip_constant=True)
        dtt._fit(dt)

        output = dtt._reverse_transform(np.array([None]))

        expected = pd.Series(pd.to_datetime(['NaT']))
        pd.testing.assert_series_equal(output, expected)


class TestDatetimeRoundedTransformer:

    def test___init___strip_is_true(self):
        """Test that by default the ``strip_constant`` is set to True."""
        dtrt = DatetimeRoundedTransformer()

        # assert
        assert dtrt.strip_constant
