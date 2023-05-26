import numpy as np
import pandas as pd

from rdt.transformers.datetime import OptimizedTimestampEncoder, UnixTimestampEncoder


class TestUnixTimestampEncoder:
    def test_unixtimestampencoder(self):
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
        """Test that transformed data will contain null values."""
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
            'column': [None, 845510400000000000, -145497600000000000]
        })
        pd.testing.assert_frame_equal(expect_transformed, transformed)
        pd.testing.assert_frame_equal(reverted, data)

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
