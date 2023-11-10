import numpy as np
import pandas as pd

from rdt.transformers.datetime import OptimizedTimestampEncoder, UnixTimestampEncoder


class TestUnixTimestampEncoder:
    def test_unixtimestampencoder(self):
        """Test the ``UnixTimestampEncoder`` end to end."""
        # Setup
        ute = UnixTimestampEncoder(missing_value_replacement='mean')
        data = pd.DataFrame({'column': pd.to_datetime([None, '1996-10-17', '1965-05-23'])})

        # Run
        ute.fit(data, column='column')
        ute.set_random_state(np.random.RandomState(7), 'reverse_transform')
        transformed = ute.transform(data)
        reverted = ute.reverse_transform(transformed)

        # Asserts
        expected_transformed = pd.DataFrame({
            'column': [3.500064e+17, 845510400000000000, -145497600000000000]
        })

        pd.testing.assert_frame_equal(expected_transformed, transformed)
        pd.testing.assert_frame_equal(reverted, data)

    def test_unixtimestampencoder_different_format(self):
        """Test the ``UnixTimestampEncoder`` with a unique datetime format."""
        ute = UnixTimestampEncoder(missing_value_replacement='mean', datetime_format='%b %d, %Y')
        data = pd.DataFrame({'column': [None, 'Oct 17, 1996', 'May 23, 1965']})

        # Run
        ute.fit(data, column='column')
        ute.set_random_state(np.random.RandomState(7), 'reverse_transform')
        transformed = ute.transform(data)
        reverted = ute.reverse_transform(transformed)

        # Asserts
        expect_transformed = pd.DataFrame({
            'column': [3.500064e+17, 845510400000000000, -145497600000000000]
        })
        pd.testing.assert_frame_equal(expect_transformed, transformed)
        pd.testing.assert_frame_equal(reverted, data)

    def test_unixtimestampencoder_with_missing_value_generation_none(self):
        """Test that transformed data will replace nans with the mean."""
        # Setup
        ute = UnixTimestampEncoder(
            missing_value_replacement='mean',
            missing_value_generation=None,
            datetime_format='%b %d, %Y'
        )
        data = pd.DataFrame({'column': [None, 'Oct 17, 1996', 'May 23, 1965']})

        # Run
        ute.fit(data, column='column')
        ute.set_random_state(np.random.RandomState(7), 'reverse_transform')
        transformed = ute.transform(data)
        reverted = ute.reverse_transform(transformed)

        # Asserts
        expect_transformed = pd.DataFrame({
            'column': [3.500064e+17, 845510400000000000, -145497600000000000]
        })
        expected_reversed = pd.DataFrame({
            'column': ['Feb 03, 1981', 'Oct 17, 1996', 'May 23, 1965']
        })
        pd.testing.assert_frame_equal(expect_transformed, transformed)
        pd.testing.assert_frame_equal(reverted, expected_reversed)

    def test_unixtimestampencoder_with_missing_value_replacement_random(self):
        """Test that transformed data will replace nans with random values from the data."""
        # Setup
        ute = UnixTimestampEncoder(
            missing_value_replacement='random',
            datetime_format='%b %d, %Y'
        )
        data = pd.DataFrame({'column': [None, 'Oct 17, 1996', 'May 23, 1965']})

        # Run
        ute.fit(data, column='column')
        ute.set_random_state(np.random.RandomState(7), 'reverse_transform')
        transformed = ute.transform(data)
        reverted = ute.reverse_transform(transformed)

        # Asserts
        expect_transformed = pd.DataFrame({
            'column': [-7.007396e+16, 845510400000000000, -145497600000000000]
        })
        expected_reversed = pd.DataFrame({
            'column': [np.nan, 'Oct 17, 1996', 'May 23, 1965']
        })
        pd.testing.assert_frame_equal(expect_transformed, transformed)
        pd.testing.assert_frame_equal(reverted, expected_reversed)

    def test_unixtimestampencoder_with_model_missing_values(self):
        """Test that `model_missing_values` is accepted by the transformer."""
        # Setup
        ute = UnixTimestampEncoder('mean', True)
        data = pd.DataFrame({'column': pd.to_datetime([None, '1996-10-17', '1965-05-23'])})

        # Run
        ute.fit(data, column='column')
        ute.set_random_state(np.random.RandomState(7), 'reverse_transform')
        transformed = ute.transform(data)
        reverted = ute.reverse_transform(transformed)

        # Asserts
        expected_transformed = pd.DataFrame({
            'column': [3.500064e+17, 845510400000000000, -145497600000000000],
            'column.is_null': [1., 0., 0.]
        })

        pd.testing.assert_frame_equal(expected_transformed, transformed)
        pd.testing.assert_frame_equal(reverted, data)

    def test_unixtimestampencoder_with_integer_datetimes(self):
        """Test that the transformer properly handles integer columns."""
        # Setup
        ute = UnixTimestampEncoder('mean', True, datetime_format='%m%d%Y')
        data = pd.DataFrame({'column': [1201992, 11022028, 10011990]})

        # Run
        ute.fit(data, column='column')
        ute.set_random_state(np.random.RandomState(7), 'reverse_transform')
        transformed = ute.transform(data)
        reverted = ute.reverse_transform(transformed)

        # Asserts
        expected_transformed = pd.DataFrame({
            'column': [6.958656e+17, 1.856736e+18, 6.547392e+17],
        })

        pd.testing.assert_frame_equal(expected_transformed, transformed)
        pd.testing.assert_frame_equal(reverted, data)

    def test_unixtimestampencoder_with_nans(self):
        """Test that the transformer properly handles null columns."""
        # Setup
        ute = UnixTimestampEncoder('mean', True)
        data = pd.DataFrame({'column': [np.nan, np.nan, np.nan]})

        # Run
        ute.fit(data, column='column')
        ute.set_random_state(np.random.RandomState(7), 'reverse_transform')
        transformed = ute.transform(data)
        reverted = ute.reverse_transform(transformed)

        # Asserts
        expected_transformed = pd.DataFrame({
            'column': [0., 0., 0.],
            'column.is_null': [1., 1., 1.]
        })

        pd.testing.assert_frame_equal(expected_transformed, transformed)
        pd.testing.assert_frame_equal(reverted, data)

    def test_with_enforce_min_max_values_true(self):
        """Test that the transformer properly clipped out of bounds values."""
        # Setup
        ute = UnixTimestampEncoder(enforce_min_max_values=True)
        data = pd.DataFrame({'column': ['Feb 03, 1981', 'Oct 17, 1996', 'May 23, 1965']})
        ute.fit(data, column='column')

        # Run
        transformed = ute.transform(data)
        min_val = transformed['column'].min()
        max_val = transformed['column'].max()
        transformed.loc[transformed['column'] == min_val, 'column'] = min_val - 1e17
        transformed.loc[transformed['column'] == max_val, 'column'] = max_val + 1e17
        reverted = ute.reverse_transform(transformed)

        # Asserts
        assert ute._min_value == min_val
        assert ute._max_value == max_val
        pd.testing.assert_frame_equal(reverted, data)


class TestOptimizedTimestampEncoder:
    def test_optimizedtimestampencoder(self):
        ote = OptimizedTimestampEncoder(missing_value_replacement='mean')
        data = pd.DataFrame({'column': pd.to_datetime([None, '1996-10-17', '1965-05-23'])})

        # Run
        ote.fit(data, column='column')
        ote.set_random_state(np.random.RandomState(7), 'reverse_transform')
        transformed = ote.transform(data)
        reverted = ote.reverse_transform(transformed)

        # Asserts
        expect_transformed = pd.DataFrame({'column': [4051.0, 9786.0, -1684.0]})
        pd.testing.assert_frame_equal(expect_transformed, transformed)
        pd.testing.assert_frame_equal(reverted, data)
