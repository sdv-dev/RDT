import numpy as np
import pandas as pd

from rdt.transformers.numerical import GaussianCopulaTransformer, NumericalTransformer


class TestNumericalTransformer:

    def test_null_column(self):
        data = pd.DataFrame([1, 2, 1, 2, np.nan, 1], columns=['a'])

        nt = NumericalTransformer()
        nt.fit(data, list(data.columns))
        transformed = nt.transform(data)

        assert isinstance(transformed, pd.DataFrame)
        assert transformed.shape == (6, 2)
        assert list(transformed.iloc[:, 1]) == [0, 0, 0, 0, 1, 0]

        reverse = nt.reverse_transform(transformed)

        np.testing.assert_array_almost_equal(reverse, data, decimal=2)

    def test_not_null_column(self):
        data = pd.DataFrame([1, 2, 1, 2, np.nan, 1], columns=['a'])

        nt = NumericalTransformer(null_column=False)
        nt.fit(data, list(data.columns))
        transformed = nt.transform(data)

        assert isinstance(transformed, pd.DataFrame)
        assert transformed.shape == (6, 1)

        reverse = nt.reverse_transform(transformed)

        np.testing.assert_array_almost_equal(reverse, data, decimal=2)

    def test_int(self):
        data = pd.DataFrame([1, 2, 1, 2, 1], columns=['a'])

        nt = NumericalTransformer(dtype=int)
        nt.fit(data, list(data.columns))
        transformed = nt.transform(data)

        assert isinstance(transformed, pd.DataFrame)
        assert transformed.shape == (5, 1)

        reverse = nt.reverse_transform(transformed)
        assert list(reverse['a']) == [1, 2, 1, 2, 1]

    def test_int_nan(self):
        data = pd.DataFrame([1, 2, 1, 2, 1, np.nan], columns=['a'])

        nt = NumericalTransformer(dtype=int)
        nt.fit(data, list(data.columns))
        transformed = nt.transform(data)

        assert isinstance(transformed, pd.DataFrame)
        assert transformed.shape == (6, 2)

        reverse = nt.reverse_transform(transformed)
        np.testing.assert_array_almost_equal(reverse, data, decimal=2)


class TestGaussianCopulaTransformer:

    def test_stats(self):
        data = pd.DataFrame(np.random.normal(loc=4, scale=4, size=1000), columns=['a'])

        ct = GaussianCopulaTransformer()
        ct.fit(data, list(data.columns))
        transformed = ct.transform(data)

        assert isinstance(transformed, pd.DataFrame)
        assert transformed.shape == (1000, 1)

        np.testing.assert_almost_equal(transformed['a.value'].mean(), 0, decimal=1)
        np.testing.assert_almost_equal(transformed['a.value'].std(), 1, decimal=1)

        reverse = ct.reverse_transform(transformed)

        np.testing.assert_array_almost_equal(reverse, data, decimal=1)

    def test_null_column(self):
        data = pd.DataFrame([1, 2, 1, 2, np.nan, 1], columns=['a'])

        ct = GaussianCopulaTransformer()
        ct.fit(data, list(data.columns))
        transformed = ct.transform(data)

        assert isinstance(transformed, pd.DataFrame)
        assert transformed.shape == (6, 2)
        assert list(transformed.iloc[:, 1]) == [0, 0, 0, 0, 1, 0]

        reverse = ct.reverse_transform(transformed)

        np.testing.assert_array_almost_equal(reverse, data, decimal=2)

    def test_not_null_column(self):
        data = pd.DataFrame([1, 2, 1, 2, np.nan, 1], columns=['a'])

        ct = GaussianCopulaTransformer(null_column=False)
        ct.fit(data, list(data.columns))
        transformed = ct.transform(data)

        assert isinstance(transformed, pd.DataFrame)
        assert transformed.shape == (6, 1)

        reverse = ct.reverse_transform(transformed)

        np.testing.assert_array_almost_equal(reverse, data, decimal=2)

    def test_int(self):
        data = pd.DataFrame([1, 2, 1, 2, 1], columns=['a'])

        ct = GaussianCopulaTransformer(dtype=int)
        ct.fit(data, list(data.columns))
        transformed = ct.transform(data)

        assert isinstance(transformed, pd.DataFrame)
        assert transformed.shape == (5, 1)

        reverse = ct.reverse_transform(transformed)
        assert list(reverse['a']) == [1, 2, 1, 2, 1]

    def test_int_nan(self):
        data = pd.DataFrame([1, 2, 1, 2, 1, np.nan], columns=['a'])

        ct = GaussianCopulaTransformer(dtype=int)
        ct.fit(data, list(data.columns))
        transformed = ct.transform(data)

        assert isinstance(transformed, pd.DataFrame)
        assert transformed.shape == (6, 2)

        reverse = ct.reverse_transform(transformed)
        np.testing.assert_array_almost_equal(reverse, data, decimal=2)
