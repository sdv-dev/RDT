import numpy as np
import pandas as pd

from tests.performance.datasets import categorical


class TestRandomIntegerGenerator:

    def test(self):
        output = categorical.RandomIntegerGenerator.generate(10)
        assert len(output) == 10
        assert output.dtype == np.int
        assert len(pd.unique(output)) < 6
        assert np.isnan(output).sum() == 0


class TestRandomIntegerNaNsGenerator:

    def test(self):
        output = categorical.RandomIntegerNaNsGenerator.generate(10)
        assert len(output) == 10
        assert output.dtype == np.float
        assert len(pd.unique(output)) < 7
        assert np.isnan(output).sum() > 0


class TestRandomCategoricalGenerator:

    def test(self):
        output = categorical.RandomCategoricalGenerator.generate(10)
        assert len(output) == 10
        assert output.dtype.type == np.str_
        assert len(pd.unique(output)) < 6
        assert pd.isnull(output).sum() == 0


class TestRandomCategoricalNaNsGenerator:

    def test(self):
        output = categorical.RandomCategoricalNaNsGenerator.generate(10)
        assert len(output) == 10
        assert output.dtype.type == np.object_
        assert len(pd.unique(output)) < 7
        assert sum(pd.isnull(output)) > 0


class TestSingleIntegerGenerator:

    def test(self):
        output = categorical.SingleIntegerGenerator.generate(10)
        assert len(output) == 10
        assert output.dtype == np.int
        assert len(pd.unique(output)) == 1
        assert np.isnan(output).sum() == 0


class TestSingleIntegerNaNsGenerator:

    def test(self):
        output = categorical.SingleIntegerNaNsGenerator.generate(10)
        assert len(output) == 10
        assert output.dtype == np.float
        assert len(pd.unique(output)) == 2
        assert np.isnan(output).sum() >= 1


class TestSingleCategoricalGenerator:

    def test(self):
        output = categorical.SingleCategoricalGenerator.generate(10)
        assert len(output) == 10
        assert output.dtype.type == np.str_
        assert len(pd.unique(output)) == 1
        assert pd.isnull(output).sum() == 0


class TestSingleCategoricalNaNsGenerator:

    def test(self):
        output = categorical.SingleCategoricalNaNsGenerator.generate(10)
        assert len(output) == 10
        assert output.dtype.type == np.object_
        assert len(pd.unique(output)) == 2
        assert sum(pd.isnull(output)) >= 1


class TestUniqueIntegerGenerator:

    def test(self):
        output = categorical.UniqueIntegerGenerator.generate(10)
        assert len(output) == 10
        assert output.dtype == np.int
        assert len(pd.unique(output)) == 10
        assert np.isnan(output).sum() == 0


class TestUniqueIntegerNaNsGenerator:

    def test(self):
        output = categorical.UniqueIntegerNaNsGenerator.generate(10)
        nulls = np.isnan(output).sum()

        assert len(output) == 10
        assert output.dtype == np.float
        assert len(pd.unique(output)) == 10 - nulls + 1
        assert nulls > 0


class TestUniqueCategoricalGenerator:

    def test(self):
        output = categorical.UniqueCategoricalGenerator.generate(10)
        assert len(output) == 10
        assert output.dtype.type == np.str_
        assert len(pd.unique(output)) == 10
        assert pd.isnull(output).sum() == 0


class TestUniqueCategoricalNaNsGenerator:

    def test(self):
        output = categorical.UniqueCategoricalNaNsGenerator.generate(10)
        nulls = sum(pd.isnull(output))

        assert len(output) == 10
        assert output.dtype == np.object_
        assert len(pd.unique(output)) == 10 - nulls + 1
        assert nulls > 0
