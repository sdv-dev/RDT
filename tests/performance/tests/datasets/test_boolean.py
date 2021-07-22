import pandas as pd

from tests.performance.datasets import boolean


class TestRandomBooleanGenerator:

    def test(self):
        output = boolean.RandomBooleanGenerator.generate(10)
        assert len(output) == 10
        assert output.dtype == bool
        assert len(pd.unique(output)) == 2
        assert pd.isnull(output).sum() == 0


class TestRandomBooleanNaNsGenerator:

    def test(self):
        output = boolean.RandomBooleanNaNsGenerator.generate(10)
        assert len(output) == 10
        assert output.dtype == 'O'
        assert len(pd.unique(output)) == 3
        assert pd.isnull(output).sum() > 0


class TestRandomSkewedBooleanGenerator:

    def test(self):
        output = boolean.RandomSkewedBooleanGenerator.generate(10)
        assert len(output) == 10
        assert output.dtype == bool
        assert len(pd.unique(output)) == 2
        assert pd.isnull(output).sum() == 0


class TestRandomSkewedBooleanNaNsGenerator:

    def test(self):
        output = boolean.RandomSkewedBooleanNaNsGenerator.generate(10)
        assert len(output) == 10
        assert output.dtype == 'O'
        assert len(pd.unique(output)) == 3
        assert pd.isnull(output).sum() > 0


class TestConstantBooleanGenerator:

    def test(self):
        output = boolean.ConstantBooleanGenerator.generate(10)
        assert len(output) == 10
        assert output.dtype == bool
        assert len(pd.unique(output)) == 1
        assert pd.isnull(output).sum() == 0


class TestConstantBooleanNaNsGenerator:

    def test(self):
        output = boolean.ConstantBooleanNaNsGenerator.generate(10)
        assert len(output) == 10
        assert output.dtype == 'O'
        assert len(pd.unique(output)) == 2
        assert pd.isnull(output).sum() > 0
