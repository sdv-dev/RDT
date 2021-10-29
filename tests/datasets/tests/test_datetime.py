import datetime as dt

import numpy as np
import pandas as pd

from tests.datasets import datetime


class TestRandomGapDatetimeGenerator:

    def test(self):
        output = datetime.RandomGapDatetimeGenerator.generate(10)
        assert len(output) == 10
        assert output.dtype == 'datetime64[us]'
        assert len(pd.unique(output)) > 1
        assert np.isnan(output).sum() == 0


class TestRandomGapSecondsDatetimeGenerator:

    def test(self):
        output = datetime.RandomGapSecondsDatetimeGenerator.generate(10)
        assert len(output) == 10
        assert output.dtype == 'datetime64[us]'
        assert len(pd.unique(output)) > 1
        assert np.isnan(output).sum() == 0


class TestRandomGapDatetimeNaNsGenerator:

    def test(self):
        output = datetime.RandomGapDatetimeNaNsGenerator.generate(10)
        assert len(output) == 10
        assert output.dtype == 'O'
        assert len(pd.unique(output)) > 1
        assert pd.isna(output).sum() > 0


class TestEqualGapHoursDatetimeGenerator:

    def test(self):
        output = datetime.EqualGapHoursDatetimeGenerator.generate(10)
        assert len(output) == 10
        assert output.dtype == 'datetime64[us]'
        assert all(x == dt.timedelta(hours=1) for x in np.diff(output))
        assert np.isnan(output).sum() == 0


class TestEqualGapDaysDatetimeGenerator:

    def test(self):
        output = datetime.EqualGapDaysDatetimeGenerator.generate(10)
        assert len(output) == 10
        assert output.dtype == 'datetime64[us]'
        assert all(x == dt.timedelta(1) for x in np.diff(output))
        assert np.isnan(output).sum() == 0


class TestEqualGapWeeksDatetimeGenerator:

    def test(self):
        output = datetime.EqualGapWeeksDatetimeGenerator.generate(10)
        assert len(output) == 10
        assert output.dtype == 'datetime64[us]'
        assert all(x == dt.timedelta(weeks=1) for x in np.diff(output))
        assert np.isnan(output).sum() == 0
