import numpy as np
import pandas as pd

from tests.datasets import numerical


class TestRandomIntegerGenerator:

    def test(self):
        output = numerical.RandomIntegerGenerator.generate(10)
        assert len(output) == 10
        assert output.dtype == int
        assert len(pd.unique(output)) > 1
        assert np.isnan(output).sum() == 0


class TestRandomIntegerNaNsGenerator:

    def test(self):
        output = numerical.RandomIntegerNaNsGenerator.generate(10)
        assert len(output) == 10
        assert output.dtype == float
        assert len(pd.unique(output)) > 1
        assert np.isnan(output).sum() > 0


class TestConstantIntegerGenerator:

    def test(self):
        output = numerical.ConstantIntegerGenerator.generate(10)
        assert len(output) == 10
        assert output.dtype == int
        assert len(pd.unique(output)) == 1
        assert np.isnan(output).sum() == 0


class TestConstantIntegerNaNsGenerator:

    def test(self):
        output = numerical.ConstantIntegerNaNsGenerator.generate(10)
        assert len(output) == 10
        assert output.dtype == float
        assert len(pd.unique(output)) == 2
        assert np.isnan(output).sum() >= 1


class TestAlmostConstantIntegerGenerator:

    def test(self):
        output = numerical.AlmostConstantIntegerGenerator.generate(10)
        assert len(output) == 10
        assert output.dtype == int
        assert len(pd.unique(output)) == 2
        assert np.isnan(output).sum() == 0


class TestAlmostConstantIntegerNaNsGenerator:

    def test(self):
        output = numerical.AlmostConstantIntegerNaNsGenerator.generate(10)
        assert len(output) == 10
        assert output.dtype == float
        assert len(pd.unique(output)) == 3
        assert np.isnan(output).sum() >= 1


class TestNormalGenerator:

    def test(self):
        output = numerical.NormalGenerator.generate(10)
        assert len(output) == 10
        assert output.dtype == float
        assert len(pd.unique(output)) == 10
        assert np.isnan(output).sum() == 0


class TestNormalNaNsGenerator:

    def test(self):
        output = numerical.NormalNaNsGenerator.generate(10)
        assert len(output) == 10
        assert output.dtype == float
        assert 1 < len(pd.unique(output)) <= 10
        assert np.isnan(output).sum() >= 1


class TestBigNormalGenerator:

    def test(self):
        output = numerical.BigNormalGenerator.generate(10)
        assert len(output) == 10
        assert output.dtype == float
        assert len(pd.unique(output)) == 10
        assert np.isnan(output).sum() == 0


class TestBigNormalNaNsGenerator:

    def test(self):
        output = numerical.BigNormalNaNsGenerator.generate(10)
        assert len(output) == 10
        assert output.dtype == float
        assert 1 < len(pd.unique(output)) <= 10
        assert np.isnan(output).sum() >= 1
