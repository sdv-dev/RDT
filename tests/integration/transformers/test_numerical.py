import numpy as np
import pandas as pd
from copulas import univariate

from rdt.transformers.null import NullTransformer
from rdt.transformers.numerical import (
    ClusterBasedNormalizer,
    FloatFormatter,
    GaussianNormalizer,
    LogScaler,
)


class TestFloatFormatter:
    def test_missing_value_generation_from_column(self):
        """Test end to end with ``missing_value_generation`` set to ``from_column``.

        The transform method should create a boolean column for the missing values.
        """
        data = pd.DataFrame([1, 2, 1, 2, np.nan, 1], columns=['a'])
        column = 'a'

        nt = FloatFormatter(
            missing_value_replacement='mean',
            missing_value_generation='from_column',
        )
        nt.fit(data, column)
        transformed = nt.transform(data)

        assert isinstance(transformed, pd.DataFrame)
        assert transformed.shape == (6, 2)
        assert list(transformed.iloc[:, 1]) == [0, 0, 0, 0, 1, 0]

        reverse = nt.reverse_transform(transformed)

        np.testing.assert_array_almost_equal(reverse, data, decimal=2)

    def test_int(self):
        """Test end to end on a column of all ints."""
        data = pd.DataFrame([1, 2, 1, 2, 1], columns=['a'])
        column = 'a'

        nt = FloatFormatter()
        nt.fit(data, column)
        transformed = nt.transform(data)

        assert isinstance(transformed, pd.DataFrame)
        assert transformed.shape == (5, 1)

        reverse = nt.reverse_transform(transformed)
        assert list(reverse['a']) == [1, 2, 1, 2, 1]

    def test_int_nan_default_missing_value_generation(self):
        """Test that NaNs are randomly inserted in the output."""
        data = pd.DataFrame([1, 2, 1, 2, 1, np.nan], columns=['a'])
        column = 'a'

        nt = FloatFormatter()
        nt.fit(data, column)
        transformed = nt.transform(data)

        assert isinstance(transformed, pd.DataFrame)
        assert transformed.shape == (6, 1)

        reverse = nt.reverse_transform(transformed)
        assert len(reverse) == 6
        assert reverse['a'][5] == 1.4 or np.isnan(reverse['a'][5])
        for value in reverse['a'][:5]:
            assert value in {1, 2} or np.isnan(value)

    def test_computer_representation(self):
        """Test that the ``computer_representation`` is learned and applied on the output."""
        data = pd.DataFrame([1, 2, 1, 2, 1], columns=['a'])
        column = 'a'

        nt = FloatFormatter(computer_representation='Int8')
        nt.fit(data, column)
        transformed = nt.transform(data)

        assert isinstance(transformed, pd.DataFrame)
        assert transformed.shape == (5, 1)

        reverse = nt.reverse_transform(transformed)
        assert list(reverse['a']) == [1, 2, 1, 2, 1]

    def test_missing_value_generation_none(self):
        """Test when ``missing_value_generation`` is ``None``.

        When ``missing_value_generation`` is ``None`` the NaNs should be replaced by the mean.
        """
        # Setup
        data = pd.DataFrame([1, 2, 1, 2, 1, np.nan], columns=['a'])
        column = 'a'
        fft = FloatFormatter(missing_value_generation=None)
        fft.fit(data, column)

        # Run
        transformed = fft.transform(data)

        # Assert
        assert isinstance(transformed, pd.DataFrame)
        assert transformed.shape == (6, 1)
        assert transformed['a'].iloc[5] == 1.4

    def test_model_missing_value(self):
        """Test that we are still able to use ``model_missing_value``."""
        # Setup
        data = pd.DataFrame([1, 2, 1, 2, np.nan, 1], columns=['a'])
        column = 'a'

        # Run
        nt = FloatFormatter('mean', True)
        nt.fit(data, column)
        transformed = nt.transform(data)
        reverse = nt.reverse_transform(transformed)

        # Assert
        assert isinstance(transformed, pd.DataFrame)
        assert transformed.shape == (6, 2)
        assert list(transformed.iloc[:, 1]) == [0, 0, 0, 0, 1, 0]
        np.testing.assert_array_almost_equal(reverse, data, decimal=2)

    def test_missing_value_replacement_set_to_random_and_model_missing_values(
        self,
    ):
        """Test that we are still able to use ``missing_value_replacement`` when is ``random``."""
        # Setup
        data = pd.DataFrame({'a': [1, 2, 3, np.nan, np.nan, 4]})

        # Run
        ft = FloatFormatter('random', True)
        ft.fit(data, 'a')
        transformed = ft.transform(data)
        reverse = ft.reverse_transform(transformed)

        # Assert
        expected_transformed = pd.DataFrame({
            'a': [1.0, 2.0, 3.0, 3.465976493452848, 1.5297519377926643, 4.0],
            'a.is_null': [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        })
        pd.testing.assert_frame_equal(transformed, expected_transformed)
        pd.testing.assert_frame_equal(reverse, data)
        np.testing.assert_array_almost_equal(reverse, data, decimal=2)

    def test_missing_value_replacement_random_all_nans(self):
        """Test ``FloatFormatter`` with all ``nans``.

        Test that ``FloatFormatter`` works when the ``missing_value_replacement`` is set to
        ``random`` and the data is all ``np.nan``.
        """
        # Setup
        data = pd.DataFrame({'a': [np.nan] * 10})
        ft = FloatFormatter('random')

        # Run
        ft.fit(data, 'a')
        transformed = ft.transform(data)
        reverse_transformed = ft.reverse_transform(transformed)

        # Assert
        expected_transformed = pd.DataFrame({'a': [0.0] * 10})
        expected_reverse_transformed = pd.DataFrame({'a': [np.nan] * 10})
        pd.testing.assert_frame_equal(transformed, expected_transformed)
        pd.testing.assert_frame_equal(reverse_transformed, expected_reverse_transformed)

    def test__reverse_transform_from_manually_set_parameters(self):
        """Test the ``reverse_transform`` after manually setting parameters."""
        # Setup
        data = pd.DataFrame({'column_name': pd.Series([1, 2, 1, 3, 12, 9, 8], dtype='int64')})
        transformed = pd.DataFrame({
            'column_name': [1.000, 2.000, 1.000, 3.000, 12.000, 9.000, 8.000]
        })
        transformer = FloatFormatter()
        column_name = 'column_name'
        null_transformer = NullTransformer('mean')
        min_max_value = (0.0, 100.0)
        rounding_digits = 3
        dtype = 'int64'

        # Run
        transformer._set_fitted_parameters(
            column_name=column_name,
            null_transformer=null_transformer,
            rounding_digits=rounding_digits,
            min_max_values=min_max_value,
            dtype=dtype,
        )
        output = transformer.reverse_transform(transformed)

        # Assert
        pd.testing.assert_series_equal(output['column_name'], data['column_name'])

    def test__reverse_transform_from_manually_set_parameters_from_column(self):
        """Test the ``reverse_transform`` after manually setting parameters."""
        # Setup
        data = pd.DataFrame({
            'column_name': pd.Series([1, 2, np.nan, 3, 12, np.nan, 8], dtype='Int64')
        })
        transformed = pd.DataFrame({
            'column_name': [1.000, 2.000, 1.000, 3.000, 12.000, 9.000, 8.000],
            'column_name.is_null': [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        })
        transformer = FloatFormatter()
        column_name = 'column_name'
        null_transformer = NullTransformer('mean', 'from_column')
        null_transformer._set_fitted_parameters(0.2)
        min_max_value = (0.0, 100.0)
        rounding_digits = 3
        dtype = 'Int64'

        # Run
        transformer._set_fitted_parameters(
            column_name=column_name,
            null_transformer=null_transformer,
            rounding_digits=rounding_digits,
            min_max_values=min_max_value,
            dtype=dtype,
        )
        output = transformer.reverse_transform(transformed)

        # Assert
        pd.testing.assert_series_equal(output['column_name'], data['column_name'])

    def test__reverse_transform_from_manually_set_parameters_random(self):
        """Test the ``reverse_transform`` after manually setting parameters."""
        # Setup
        data = pd.DataFrame({'column_name': pd.Series([1, 2, 1, 3, 12, 9, 8, 4], dtype='Int64')})
        transformed = pd.DataFrame({
            'column_name': [1.000, 2.000, 1.000, 3.000, 12.000, 9.000, 8.000, 4.000]
        })
        transformer = FloatFormatter()
        column_name = 'column_name'
        null_transformer = NullTransformer('mean', 'random')
        null_transformer._set_fitted_parameters(0.2)
        min_max_value = (0.0, 100.0)
        rounding_digits = 3
        dtype = 'Int64'

        # Run
        transformer._set_fitted_parameters(
            column_name=column_name,
            null_transformer=null_transformer,
            rounding_digits=rounding_digits,
            min_max_values=min_max_value,
            dtype=dtype,
        )
        output = transformer.reverse_transform(transformed)
        nan_indices = output[output.isna().any(axis=1)].index
        compare_data = data.drop(index=nan_indices)
        compare_output = output.drop(index=nan_indices)

        # Assert
        pd.testing.assert_series_equal(compare_output['column_name'], compare_data['column_name'])

    def test__support__nullable_numerical_pandas_dtypes(self):
        """Test that the transformer supports the nullable numerical pandas dtypes."""
        # Setup
        data = pd.DataFrame({
            'Int8': pd.Series([1, 2, -3, pd.NA, None, pd.NA], dtype='Int8'),
            'Int16': pd.Series([1, 2, -3, pd.NA, None, pd.NA], dtype='Int16'),
            'Int32': pd.Series([1, 2, -3, pd.NA, None, pd.NA], dtype='Int32'),
            'Int64': pd.Series([1, 2, -3, pd.NA, None, pd.NA], dtype='Int64'),
            'Float32': pd.Series([1.123, 2.23, 3.3, pd.NA, None, pd.NA], dtype='Float32'),
            'Float64': pd.Series([1.1234, 2.234, 3.33, pd.NA, None, pd.NA], dtype='Float64'),
        })
        expected_rounding_digits = {
            'Int8': 0,
            'Int16': 0,
            'Int32': 0,
            'Int64': 0,
            'Float32': 3,
            'Float64': 4,
        }

        # Run and Assert
        for column in data.columns:
            ff = FloatFormatter(learn_rounding_scheme=True, computer_representation=column)
            ff.fit(data, column)
            transformed = ff.transform(data)
            reverse_transformed = ff.reverse_transform(transformed)

            assert transformed[column].dtype == 'float64'
            assert reverse_transformed[column].dtype == data[column].dtype
            assert reverse_transformed[column].isna().any()
            assert ff._rounding_digits == expected_rounding_digits[column]
            pd.testing.assert_series_equal(
                reverse_transformed[column],
                reverse_transformed[column].round(expected_rounding_digits[column]),
            )

    def test__set_fitted_parameter_rounding_to_integer(self):
        """Test the ``_set_fitted_parameters`` method with rounding_digits set to 0."""
        # Setup
        data = pd.DataFrame({
            'col 1': 100 * np.random.random(10),
        })
        transformer = FloatFormatter()

        # Run
        transformer._set_fitted_parameters(
            column_name='col 1',
            null_transformer=NullTransformer(),
            rounding_digits=0,
            dtype='float',
        )
        reverse_transformed_data = transformer.reverse_transform(data)

        # Assert
        pd.testing.assert_frame_equal(reverse_transformed_data, data.round(0))


class TestGaussianNormalizer:
    def test_stats(self):
        data = pd.DataFrame(np.random.normal(loc=4, scale=4, size=1000), columns=['a'])
        column = 'a'

        ct = GaussianNormalizer()
        ct.fit(data, column)
        transformed = ct.transform(data)

        assert isinstance(transformed, pd.DataFrame)
        assert transformed.shape == (1000, 1)

        np.testing.assert_almost_equal(transformed['a'].mean(), 0, decimal=1)
        np.testing.assert_almost_equal(transformed['a'].std(), 1, decimal=1)

        reverse = ct.reverse_transform(transformed)

        np.testing.assert_array_almost_equal(reverse, data, decimal=1)

    def test_missing_value_generation_from_column(self):
        data = pd.DataFrame([1, 2, 1, 2, np.nan, 1], columns=['a'])
        column = 'a'

        ct = GaussianNormalizer(missing_value_generation='from_column')
        ct.fit(data, column)
        transformed = ct.transform(data)

        assert isinstance(transformed, pd.DataFrame)
        assert transformed.shape == (6, 2)
        assert list(transformed.iloc[:, 1]) == [0, 0, 0, 0, 1, 0]

        reverse = ct.reverse_transform(transformed)

        np.testing.assert_array_almost_equal(reverse, data, decimal=2)

    def test_missing_value_generation_random(self):
        data = pd.DataFrame([1, 2, 1, 2, np.nan, 1], columns=['a'])
        column = 'a'

        ct = GaussianNormalizer(missing_value_generation='random')
        ct.fit(data, column)
        transformed = ct.transform(data)

        assert isinstance(transformed, pd.DataFrame)
        assert transformed.shape == (6, 1)

        reverse = ct.reverse_transform(transformed)
        expected = pd.DataFrame(
            [1.0, 1.9999999510423996, 1.0, 1.9999999510423996, 1.4, np.nan],
            columns=['a'],
        )
        pd.testing.assert_frame_equal(reverse, expected)

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

        ct = GaussianNormalizer(missing_value_generation='from_column')
        ct.fit(data, column)
        transformed = ct.transform(data)

        assert isinstance(transformed, pd.DataFrame)
        assert transformed.shape == (6, 2)

        reverse = ct.reverse_transform(transformed)
        np.testing.assert_array_almost_equal(reverse, data, decimal=2)

    def test_uniform(self):
        """Test it works when distribution='uniform'."""
        # Setup
        data = pd.DataFrame(np.random.uniform(size=1000), columns=['a'])
        ct = GaussianNormalizer(distribution='uniform')

        # Run
        ct.fit(data, 'a')
        transformed = ct.transform(data)
        reverse = ct.reverse_transform(transformed)

        # Assert
        assert isinstance(transformed, pd.DataFrame)
        assert transformed.shape == (1000, 1)

        np.testing.assert_almost_equal(transformed['a'].mean(), 0, decimal=1)
        np.testing.assert_almost_equal(transformed['a'].std(), 1, decimal=1)

        np.testing.assert_array_almost_equal(reverse, data, decimal=1)

    def test_uniform_object(self):
        """Test it works when distribution=UniformUnivariate()."""
        # Setup
        data = pd.DataFrame(np.random.uniform(size=1000), columns=['a'])
        ct = GaussianNormalizer(distribution=univariate.UniformUnivariate())

        # Run
        ct.fit(data, 'a')
        transformed = ct.transform(data)
        reverse = ct.reverse_transform(transformed)

        # Assert
        assert isinstance(transformed, pd.DataFrame)
        assert transformed.shape == (1000, 1)

        np.testing.assert_almost_equal(transformed['a'].mean(), 0, decimal=1)
        np.testing.assert_almost_equal(transformed['a'].std(), 1, decimal=1)

        np.testing.assert_array_almost_equal(reverse, data, decimal=1)

    def test_uniform_class(self):
        """Test it works when distribution=UniformUnivariate."""
        # Setup
        data = pd.DataFrame(np.random.uniform(size=1000), columns=['a'])
        ct = GaussianNormalizer(distribution=univariate.UniformUnivariate)

        # Run
        ct.fit(data, 'a')
        transformed = ct.transform(data)
        reverse = ct.reverse_transform(transformed)

        # Assert
        assert isinstance(transformed, pd.DataFrame)
        assert transformed.shape == (1000, 1)

        np.testing.assert_almost_equal(transformed['a'].mean(), 0, decimal=1)
        np.testing.assert_almost_equal(transformed['a'].std(), 1, decimal=1)

        np.testing.assert_array_almost_equal(reverse, data, decimal=1)


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
        mask = np.random.choice([1, 0], data.shape, p=[0.1, 0.9]).astype(bool)
        data[mask] = np.nan
        column = 'col'

        bgmm_transformer = ClusterBasedNormalizer(missing_value_generation='from_column')
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

    def test_out_of_bounds_reverse_transform(self):
        """Test that the reverse transform works when the data is out of bounds GH#672."""
        # Setup
        data = pd.DataFrame({
            'col': [round(i, 2) for i in np.random.uniform(0, 10, size=100)] + [None]
        })
        reverse_data = pd.DataFrame(
            data={
                'col.normalized': np.random.uniform(-10, 10, size=100),
                'col.component': np.random.choice([0.0, 1.0, 2.0, 10.0], size=100),
            }
        )
        transformer = ClusterBasedNormalizer()

        # Run
        transformer.fit(data, 'col')
        reverse = transformer.reverse_transform(reverse_data)

        # Assert
        assert isinstance(reverse, pd.DataFrame)


class TestLogScaler:
    def test_learn_rounding(self):
        """Test that transformer learns rounding scheme from data."""
        # Setup
        data = pd.DataFrame({'test': [1.0, np.nan, 1.5]})
        transformer = LogScaler(
            missing_value_generation=None,
            missing_value_replacement='mean',
            learn_rounding_scheme=True,
        )
        expected = pd.DataFrame({'test': [1.0, 1.2, 1.5]})

        # Run
        transformer.fit(data, 'test')
        transformed = transformer.transform(data)
        reversed_values = transformer.reverse_transform(transformed)

        # Assert
        np.testing.assert_array_equal(reversed_values, expected)

    def test_missing_value_generation_from_column(self):
        """Test from_column missing value generation with nans present."""
        # Setup
        data = pd.DataFrame({'test': [1.0, np.nan, 1.5]})
        transformer = LogScaler(
            missing_value_generation='from_column',
            missing_value_replacement='mean',
        )

        # Run
        transformer.fit(data, 'test')
        transformed = transformer.transform(data)
        reversed_values = transformer.reverse_transform(transformed)

        # Assert
        np.testing.assert_array_equal(reversed_values, data)

    def test_missing_value_generation_random(self):
        """Test random missing_value_generation with nans present."""
        # Setup
        data = pd.DataFrame({'test': [1.0, np.nan, 1.5, 1.5]})
        transformer = LogScaler(
            missing_value_generation='random',
            missing_value_replacement='mode',
            invert=True,
            constant=3.0,
        )
        expected = pd.DataFrame({'test': [np.nan, 1.5, 1.5, 1.5]})

        # Run
        transformer.fit(data, 'test')
        transformed = transformer.transform(data)
        reversed_values = transformer.reverse_transform(transformed)

        # Assert
        np.testing.assert_array_equal(reversed_values, expected)
