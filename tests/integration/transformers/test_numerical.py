import numpy as np

from rdt.transformers.numerical import GaussianCopulaTransformer, NumericalTransformer


class TestNumericalTransformer:

    def test_null_column(self):
        data = np.array([1, 2, 1, 2, np.nan, 1])

        nt = NumericalTransformer()
        transformed = nt.fit_transform(data)

        assert isinstance(transformed, np.ndarray)
        assert transformed.shape == (6, 2)
        assert list(transformed[:, 1]) == [0, 0, 0, 0, 1, 0]

        reverse = nt.reverse_transform(transformed)

        np.testing.assert_array_almost_equal(reverse, data, decimal=2)

    def test_not_null_column(self):
        data = np.array([1, 2, 1, 2, np.nan, 1])

        nt = NumericalTransformer(null_column=False)
        transformed = nt.fit_transform(data)

        assert isinstance(transformed, np.ndarray)
        assert transformed.shape == (6, )

        reverse = nt.reverse_transform(transformed)

        np.testing.assert_array_almost_equal(reverse, data, decimal=2)

    def test_int(self):
        data = np.array([1, 2, 1, 2, 1])

        nt = NumericalTransformer(dtype=int)
        transformed = nt.fit_transform(data)

        assert isinstance(transformed, np.ndarray)
        assert transformed.shape == (5, )

        reverse = nt.reverse_transform(transformed)
        assert list(reverse) == [1, 2, 1, 2, 1]

    def test_int_nan(self):
        data = np.array([1, 2, 1, 2, 1, np.nan])

        nt = NumericalTransformer(dtype=int)
        transformed = nt.fit_transform(data)

        assert isinstance(transformed, np.ndarray)
        assert transformed.shape == (6, 2)

        reverse = nt.reverse_transform(transformed)
        np.testing.assert_array_almost_equal(reverse, data, decimal=2)


class TestGaussianCopulaTransformer:

    def test_stats(self):
        data = np.random.normal(loc=4, scale=4, size=1000)

        ct = GaussianCopulaTransformer()
        transformed = ct.fit_transform(data)

        assert isinstance(transformed, np.ndarray)
        assert transformed.shape == (1000, )

        np.testing.assert_almost_equal(transformed.mean(), 0, decimal=1)
        np.testing.assert_almost_equal(transformed.std(), 1, decimal=1)

        reverse = ct.reverse_transform(transformed)

        np.testing.assert_array_almost_equal(reverse, data, decimal=1)

    def test_null_column(self):
        data = np.array([1, 2, 1, 2, np.nan, 1])

        ct = GaussianCopulaTransformer()
        transformed = ct.fit_transform(data)

        assert isinstance(transformed, np.ndarray)
        assert transformed.shape == (6, 2)
        assert list(transformed[:, 1]) == [0, 0, 0, 0, 1, 0]

        reverse = ct.reverse_transform(transformed)

        np.testing.assert_array_almost_equal(reverse, data, decimal=2)

    def test_not_null_column(self):
        data = np.array([1, 2, 1, 2, np.nan, 1])

        ct = GaussianCopulaTransformer(null_column=False)
        transformed = ct.fit_transform(data)

        assert isinstance(transformed, np.ndarray)
        assert transformed.shape == (6, )

        reverse = ct.reverse_transform(transformed)

        np.testing.assert_array_almost_equal(reverse, data, decimal=2)

    def test_int(self):
        data = np.array([1, 2, 1, 2, 1])

        ct = GaussianCopulaTransformer(dtype=int)
        transformed = ct.fit_transform(data)

        assert isinstance(transformed, np.ndarray)
        assert transformed.shape == (5, )

        reverse = ct.reverse_transform(transformed)
        assert list(reverse) == [1, 2, 1, 2, 1]

    def test_int_nan(self):
        data = np.array([1, 2, 1, 2, 1, np.nan])

        ct = GaussianCopulaTransformer(dtype=int)
        transformed = ct.fit_transform(data)

        assert isinstance(transformed, np.ndarray)
        assert transformed.shape == (6, 2)

        reverse = ct.reverse_transform(transformed)
        np.testing.assert_array_almost_equal(reverse, data, decimal=2)
