import numpy as np
import pandas as pd

from tests.datasets import categorical


class TestRandomIntegerGenerator:

    def test(self):
        output = categorical.RandomIntegerGenerator.generate(10)
        assert len(output) == 10
        assert output.dtype == int
        assert len(pd.unique(output)) < 6
        assert np.isnan(output).sum() == 0


class TestRandomIntegerNaNsGenerator:

    def test(self):
        output = categorical.RandomIntegerNaNsGenerator.generate(10)
        assert len(output) == 10
        assert output.dtype == float
        assert len(pd.unique(output)) < 7
        assert np.isnan(output).sum() > 0


class TestRandomStringGenerator:

    def test(self):
        output = categorical.RandomStringGenerator.generate(10)
        assert len(output) == 10
        assert output.dtype.type == np.str_
        assert len(pd.unique(output)) < 6
        assert pd.isna(output).sum() == 0


class TestRandomStringNaNsGenerator:

    def test(self):
        output = categorical.RandomStringNaNsGenerator.generate(10)
        assert len(output) == 10
        assert output.dtype.type == np.object_
        assert len(pd.unique(output)) < 7
        assert sum(pd.isna(output)) > 0


class TestRandomMixedGenerator:

    def test(self):
        output = categorical.RandomMixedGenerator.generate(10)
        assert len(output) == 10
        assert output.dtype.type == np.object_
        assert pd.isna(output).sum() == 0


class TestRandomMixedNaNsGenerator:

    def test(self):
        output = categorical.RandomMixedNaNsGenerator.generate(10)
        assert len(output) == 10
        assert output.dtype.type == np.object_
        assert sum(pd.isna(output)) > 0


class TestSingleIntegerGenerator:

    def test(self):
        output = categorical.SingleIntegerGenerator.generate(10)
        assert len(output) == 10
        assert output.dtype == int
        assert len(pd.unique(output)) == 1
        assert np.isnan(output).sum() == 0


class TestSingleIntegerNaNsGenerator:

    def test(self):
        output = categorical.SingleIntegerNaNsGenerator.generate(10)
        assert len(output) == 10
        assert output.dtype == float
        assert len(pd.unique(output)) == 2
        assert np.isnan(output).sum() >= 1


class TestSingleStringGenerator:

    def test(self):
        output = categorical.SingleStringGenerator.generate(10)
        assert len(output) == 10
        assert output.dtype.type == np.str_
        assert len(pd.unique(output)) == 1
        assert pd.isna(output).sum() == 0


class TestSingleStringNaNsGenerator:

    def test(self):
        output = categorical.SingleStringNaNsGenerator.generate(10)
        assert len(output) == 10
        assert output.dtype.type == np.object_
        assert len(pd.unique(output)) == 2
        assert sum(pd.isna(output)) >= 1


class TestUniqueIntegerGenerator:

    def test(self):
        output = categorical.UniqueIntegerGenerator.generate(10)
        assert len(output) == 10
        assert output.dtype == int
        assert len(pd.unique(output)) == 10
        assert np.isnan(output).sum() == 0


class TestUniqueIntegerNaNsGenerator:

    def test(self):
        output = categorical.UniqueIntegerNaNsGenerator.generate(10)
        nulls = np.isnan(output).sum()

        assert len(output) == 10
        assert output.dtype == float
        assert len(pd.unique(output)) == 10 - nulls + 1
        assert nulls > 0


class TestUniqueStringGenerator:

    def test(self):
        output = categorical.UniqueStringGenerator.generate(10)
        assert len(output) == 10
        assert output.dtype.type == np.str_
        assert len(pd.unique(output)) == 10
        assert pd.isna(output).sum() == 0


class TestUniqueStringNaNsGenerator:

    def test(self):
        output = categorical.UniqueStringNaNsGenerator.generate(10)
        nulls = sum(pd.isna(output))

        assert len(output) == 10
        assert output.dtype == np.object_
        assert len(pd.unique(output)) == 10 - nulls + 1
        assert nulls > 0
