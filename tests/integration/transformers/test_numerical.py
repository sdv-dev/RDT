import numpy as np
import pandas as pd

from rdt.transformers.numerical import ClusterBasedNormalizer, FloatFormatter, GaussianNormalizer


class TestFloatFormatter:

    def test_model_missing_values(self):
        data = pd.DataFrame([1, 2, 1, 2, np.nan, 1], columns=['a'])
        column = 'a'

        nt = FloatFormatter(
            missing_value_replacement='mean',
            model_missing_values=True,
        )
        nt.fit(data, column)
        transformed = nt.transform(data)

        assert isinstance(transformed, pd.DataFrame)
        assert transformed.shape == (6, 2)
        assert list(transformed.iloc[:, 1]) == [0, 0, 0, 0, 1, 0]

        reverse = nt.reverse_transform(transformed)

        np.testing.assert_array_almost_equal(reverse, data, decimal=2)

    def test_not_model_missing_values(self):
        data = pd.DataFrame([1, 2, 1, 2, np.nan, 1], columns=['a'])
        column = 'a'

        nt = FloatFormatter(model_missing_values=False)
        nt.fit(data, column)
        transformed = nt.transform(data)

        assert isinstance(transformed, pd.DataFrame)
        assert transformed.shape == (6, 1)

        reverse = nt.reverse_transform(transformed)

        np.testing.assert_array_almost_equal(reverse, data, decimal=2)

    def test_int(self):
        data = pd.DataFrame([1, 2, 1, 2, 1], columns=['a'])
        column = 'a'

        nt = FloatFormatter()
        nt.fit(data, column)
        transformed = nt.transform(data)

        assert isinstance(transformed, pd.DataFrame)
        assert transformed.shape == (5, 1)

        reverse = nt.reverse_transform(transformed)
        assert list(reverse['a']) == [1, 2, 1, 2, 1]

    def test_int_nan_not_model_missing_values(self):
        data = pd.DataFrame([1, 2, 1, 2, 1, np.nan], columns=['a'])
        column = 'a'

        nt = FloatFormatter()
        nt.fit(data, column)
        transformed = nt.transform(data)

        assert isinstance(transformed, pd.DataFrame)
        assert transformed.shape == (6, 1)

        reverse = nt.reverse_transform(transformed)
        np.testing.assert_array_almost_equal(reverse, data, decimal=2)


class TestGaussianNormalizer:

    def test_stats(self):
        data = pd.DataFrame(np.random.normal(loc=4, scale=4, size=1000), columns=['a'])
        column = 'a'

        ct = GaussianNormalizer()
        ct.fit(data, column)
        transformed = ct.transform(data)

        assert isinstance(transformed, pd.DataFrame)
        assert transformed.shape == (1000, 1)

        np.testing.assert_almost_equal(transformed['a.value'].mean(), 0, decimal=1)
        np.testing.assert_almost_equal(transformed['a.value'].std(), 1, decimal=1)

        reverse = ct.reverse_transform(transformed)

        np.testing.assert_array_almost_equal(reverse, data, decimal=1)

    def test_model_missing_values(self):
        data = pd.DataFrame([1, 2, 1, 2, np.nan, 1], columns=['a'])
        column = 'a'

        ct = GaussianNormalizer(model_missing_values=True)
        ct.fit(data, column)
        transformed = ct.transform(data)

        assert isinstance(transformed, pd.DataFrame)
        assert transformed.shape == (6, 2)
        assert list(transformed.iloc[:, 1]) == [0, 0, 0, 0, 1, 0]

        reverse = ct.reverse_transform(transformed)

        np.testing.assert_array_almost_equal(reverse, data, decimal=2)

    def test_not_model_missing_values(self):
        random_state = np.random.get_state()
        np.random.set_state(np.random.RandomState(6).get_state())
        data = pd.DataFrame([1, 2, 1, 2, np.nan, 1], columns=['a'])
        column = 'a'

        ct = GaussianNormalizer(model_missing_values=False)
        ct.fit(data, column)
        transformed = ct.transform(data)

        assert isinstance(transformed, pd.DataFrame)
        assert transformed.shape == (6, 1)

        reverse = ct.reverse_transform(transformed)
        expected = pd.DataFrame([1, 2, 1, np.nan, np.nan, 1], columns=['a'])
        pd.testing.assert_frame_equal(reverse, expected)
        np.random.set_state(random_state)

    def test_int(self):
        data = pd.DataFrame([1, 2, 1, 2, 1], columns=['a'])
        column = 'a'

        ct = GaussianNormalizer()
        ct.fit(data, column)
        transformed = ct.transform(data)

        assert isinstance(transformed, pd.DataFrame)
        assert transformed.shape == (5, 1)

        reverse = ct.reverse_transform(transformed)
        assert list(reverse['a']) == [1, 2, 1, 2, 1]

    def test_int_nan(self):
        data = pd.DataFrame([1, 2, 1, 2, 1, np.nan], columns=['a'])
        column = 'a'

        ct = GaussianNormalizer(model_missing_values=True)
        ct.fit(data, column)
        transformed = ct.transform(data)

        assert isinstance(transformed, pd.DataFrame)
        assert transformed.shape == (6, 2)

        reverse = ct.reverse_transform(transformed)
        np.testing.assert_array_almost_equal(reverse, data, decimal=2)


class TestClusterBasedNormalizer:

    def generate_data(self):
        data1 = np.random.normal(loc=5, scale=1, size=100)
        data2 = np.random.normal(loc=-5, scale=1, size=100)
        data = np.concatenate([data1, data2])

        return pd.DataFrame(data, columns=['col'])

    def test_dataframe(self):
        data = self.generate_data()
        column = 'col'

        bgmm_transformer = ClusterBasedNormalizer()
        bgmm_transformer.fit(data, column)
        transformed = bgmm_transformer.transform(data)

        assert isinstance(transformed, pd.DataFrame)
        assert transformed.shape == (200, 2)
        assert all(isinstance(x, float) for x in transformed['col.normalized'])
        assert all(isinstance(x, float) for x in transformed['col.component'])

        reverse = bgmm_transformer.reverse_transform(transformed)
        np.testing.assert_array_almost_equal(reverse, data, decimal=1)

    def test_some_nulls(self):
        random_state = np.random.get_state()
        np.random.set_state(np.random.RandomState(10).get_state())
        data = self.generate_data()
        mask = np.random.choice([1, 0], data.shape, p=[.1, .9]).astype(bool)
        data[mask] = np.nan
        column = 'col'

        bgmm_transformer = ClusterBasedNormalizer(model_missing_values=True)
        bgmm_transformer.fit(data, column)
        transformed = bgmm_transformer.transform(data)

        assert isinstance(transformed, pd.DataFrame)
        assert transformed.shape == (200, 3)
        assert all(isinstance(x, float) for x in transformed['col.normalized'])
        assert all(isinstance(x, float) for x in transformed['col.component'])
        assert all(isinstance(x, float) for x in transformed['col.is_null'])

        reverse = bgmm_transformer.reverse_transform(transformed)
        np.testing.assert_array_almost_equal(reverse, data, decimal=1)
        np.random.set_state(random_state)

    def test_data_different_sizes(self):
        data = np.concatenate([
            np.random.normal(loc=5, scale=1, size=100),
            np.random.normal(loc=100, scale=1, size=500),
        ])
        data = pd.DataFrame(data, columns=['col'])
        column = 'col'

        bgmm_transformer = ClusterBasedNormalizer()
        bgmm_transformer.fit(data, column)
        transformed = bgmm_transformer.transform(data)

        assert isinstance(transformed, pd.DataFrame)
        assert all(isinstance(x, float) for x in transformed['col.normalized'])
        assert all(isinstance(x, float) for x in transformed['col.component'])

        reverse = bgmm_transformer.reverse_transform(transformed)
        np.testing.assert_array_almost_equal(reverse, data, decimal=1)

    def test_multiple_components(self):
        random_state = np.random.get_state()
        np.random.set_state(np.random.RandomState(10).get_state())
        data = np.concatenate([
            np.random.normal(loc=5, scale=0.02, size=300),
            np.random.normal(loc=-4, scale=0.1, size=1000),
            np.random.normal(loc=-180, scale=3, size=1500),
            np.random.normal(loc=100, scale=10, size=500),
        ])
        data = pd.DataFrame(data, columns=['col'])
        data = data.sample(frac=1).reset_index(drop=True)
        column = 'col'

        bgmm_transformer = ClusterBasedNormalizer()
        bgmm_transformer.fit(data, column)
        transformed = bgmm_transformer.transform(data)

        assert isinstance(transformed, pd.DataFrame)
        assert all(isinstance(x, float) for x in transformed['col.normalized'])
        assert all(isinstance(x, float) for x in transformed['col.component'])

        reverse = bgmm_transformer.reverse_transform(transformed)
        np.testing.assert_array_almost_equal(reverse, data, decimal=1)
        np.random.set_state(random_state)
