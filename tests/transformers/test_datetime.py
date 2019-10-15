from unittest import TestCase
from unittest.mock import Mock

import numpy as np
import pandas as pd

from rdt.transformers import DatetimeTransformer


class TestDatetimeTransformer(TestCase):

    def test___init__(self):
        """Test default instance"""
        # Run
        transformer = DatetimeTransformer()

        # Asserts
        self.assertEqual(transformer.nan, 'mean', "Unexpected nan")
        self.assertIsNone(transformer.null_column, "null_column is None by default")
        self.assertIsNone(transformer.null_transformer, "null_transformer is None by default")

    def test__transform(self):
        """Test transform datetimes series to integer"""
        # Setup
        data = pd.Series([None, '1996-10-17', '1965-05-23'])
        data = pd.to_datetime(data)

        # Run
        result = DatetimeTransformer._transform(data)

        # Asserts
        expect = pd.Series([np.nan, 845510400000000000, -145497600000000000])

        pd.testing.assert_series_equal(result, expect)

    def test_fit_nan_mean_array(self):
        """Test fit nan mean with numpy.array"""
        # Setup
        data = np.array([None, '1996-10-17', '1965-05-23'])
        data = pd.to_datetime(data).to_numpy()

        # Run
        transformer = DatetimeTransformer(nan='mean')
        transformer.fit(data)

        # Asserts
        expect_nan = 'mean'
        expect_fill_value = 350006400000000000

        self.assertEqual(
            transformer.nan,
            expect_nan,
            'Unexpected nan'
        )
        self.assertEqual(
            transformer.null_transformer.fill_value,
            expect_fill_value,
            "Data mean is wrong"
        )

    def test_fit_nan_mean_series(self):
        """Test fit nan mean with pandas.Series"""
        # Setup
        data = pd.Series([None, '1996-10-17', '1965-05-23'])
        data = pd.to_datetime(data)

        # Run
        transformer = DatetimeTransformer(nan='mean')
        transformer.fit(data)

        # Asserts
        expect_nan = 'mean'
        expect_fill_value = 350006400000000000

        self.assertEqual(
            transformer.nan,
            expect_nan,
            'Unexpected nan'
        )
        self.assertEqual(
            transformer.null_transformer.fill_value,
            expect_fill_value,
            "Data mean is wrong"
        )

    def test_fit_nan_mode_array(self):
        """Test fit nan mode with numpy.array"""
        # Setup
        data = np.array([None, '1996-10-17', '1965-05-23'])
        data = pd.to_datetime(data).to_numpy()

        # Run
        transformer = DatetimeTransformer(nan='mode')
        transformer.fit(data)

        # Asserts
        expect_nan = 'mode'
        expect_fill_value = -145497600000000000

        self.assertEqual(
            transformer.nan,
            expect_nan,
            'Unexpected nan'
        )
        self.assertEqual(
            transformer.null_transformer.fill_value,
            expect_fill_value,
            "Data mean is wrong"
        )

    def test_fit_nan_mode_series(self):
        """Test fit nan mode with pandas.Series"""
        # Setup
        data = np.array([None, '1996-10-17', '1965-05-23'])
        data = pd.to_datetime(data)

        # Run
        transformer = DatetimeTransformer(nan='mode')
        transformer.fit(data)

        # Asserts
        expect_nan = 'mode'
        expect_fill_value = -145497600000000000

        self.assertEqual(
            transformer.nan,
            expect_nan,
            'Unexpected nan'
        )
        self.assertEqual(
            transformer.null_transformer.fill_value,
            expect_fill_value,
            "Data mean is wrong"
        )

    def test_fit_nan_ignore_array(self):
        """Test fit nan ignore with numpy.array"""
        # Setup
        data = np.array([None, '1996-10-17', '1965-05-23'])
        data = pd.to_datetime(data).to_numpy()

        # Run
        transformer = DatetimeTransformer(nan=None)
        transformer.fit(data)

        # Asserts
        expect_nan = None
        expect_fill_value = None

        self.assertEqual(
            transformer.nan,
            expect_nan,
            'Unexpected nan'
        )
        self.assertEqual(
            transformer.null_transformer.fill_value,
            expect_fill_value,
            "Data mean is wrong"
        )

    def test_fit_nan_ignore_series(self):
        """Test fit nan ignore with pandas.Series"""
        # Setup
        data = np.array([None, '1996-10-17', '1965-05-23'])
        data = pd.to_datetime(data)

        # Run
        transformer = DatetimeTransformer(nan=None)
        transformer.fit(data)

        # Asserts
        expect_nan = None
        expect_fill_value = None

        self.assertEqual(
            transformer.nan,
            expect_nan,
            'Unexpected nan'
        )
        self.assertEqual(
            transformer.null_transformer.fill_value,
            expect_fill_value,
            "Data mean is wrong"
        )

    def test_fit_nan_other_array(self):
        """Test fit nan custom value with numpy.array"""
        # Setup
        data = np.array([None, '1996-10-17', '1965-05-23'])
        data = pd.to_datetime(data).to_numpy()

        # Run
        transformer = DatetimeTransformer(nan=0)
        transformer.fit(data)

        # Asserts
        expect_nan = 0
        expect_fill_value = 0

        self.assertEqual(
            transformer.nan,
            expect_nan,
            'Unexpected nan'
        )
        self.assertEqual(
            transformer.null_transformer.fill_value,
            expect_fill_value,
            "Data mean is wrong"
        )

    def test_fit_nan_other_series(self):
        """Test fit nan custom value with pandas.Series"""
        # Setup
        data = np.array([None, '1996-10-17', '1965-05-23'])
        data = pd.to_datetime(data)

        # Run
        transformer = DatetimeTransformer(nan=0)
        transformer.fit(data)

        # Asserts
        expect_nan = 0
        expect_fill_value = 0

        self.assertEqual(
            transformer.nan,
            expect_nan,
            'Unexpected nan'
        )
        self.assertEqual(
            transformer.null_transformer.fill_value,
            expect_fill_value,
            "Data mean is wrong"
        )

    def test_transform_array(self):
        """Test tranform datetime arary"""
        # Setup
        data = np.array([None, '1996-10-17', '1965-05-23'])
        data = pd.to_datetime(data).to_numpy()

        data_transform = pd.Series([np.nan, 845510400000000000, -145497600000000000])

        # Run
        transformer = Mock()
        transformer._transform.return_value = data_transform

        DatetimeTransformer.transform(transformer, data)

        # Asserts
        exp_call_data = pd.Series([None, '1996-10-17', '1965-05-23'])
        expect_call_args = pd.to_datetime(exp_call_data)
        expect_call_count = 1

        pd.testing.assert_series_equal(
            transformer._transform.call_args[0][0],
            expect_call_args
        )
        self.assertEqual(
            transformer.null_transformer.transform.call_count,
            expect_call_count,
            "NullTransformer.transform must be called only once."
        )

    def test_transform_series(self):
        """Test transform datetime series"""
        # Setup
        data = pd.Series([None, '1996-10-17', '1965-05-23'])
        data = pd.to_datetime(data)

        data_transform = pd.Series([np.nan, 845510400000000000, -145497600000000000])

        # Run
        transformer = Mock()
        transformer._transform.return_value = data_transform

        DatetimeTransformer.transform(transformer, data)

        # Asserts
        exp_call_data = pd.Series([None, '1996-10-17', '1965-05-23'])
        expect_call_args = pd.to_datetime(exp_call_data)
        expect_call_count = 1

        pd.testing.assert_series_equal(
            transformer._transform.call_args[0][0],
            expect_call_args
        )
        self.assertEqual(
            transformer.null_transformer.transform.call_count,
            expect_call_count,
            "NullTransformer.transform must be called only once."
        )

    def test_reverse_transform_nan_not_ignore(self):
        """Test reverse_transform with nan not equal to ignore"""
        # Setup
        data = pd.Series([np.nan, 845510400000000000, -145497600000000000])

        reversed_data = pd.Series([np.nan, 845510400000000000, -145497600000000000])

        # Run
        transformer = Mock()
        transformer.nan = 'mean'
        transformer.null_transformer.reverse_transform.return_value = reversed_data

        DatetimeTransformer.reverse_transform(transformer, data)

        # Asserts
        expect_reverse_call_count = 1

        self.assertEqual(
            transformer.null_transformer.reverse_transform.call_count,
            expect_reverse_call_count,
            "NullTransformer.reverse_transform must be called when nan is not ignore"
        )

    def test_reverse_transform_nan_ignore(self):
        """Test reverse_transform with nan equal to ignore"""
        # Setup
        data = pd.Series([np.nan, 845510400000000000, -145497600000000000])

        # Run
        transformer = Mock()
        transformer.nan = None

        result = DatetimeTransformer.reverse_transform(transformer, data)

        # Asserts
        expect = pd.Series([
            np.nan,
            pd.to_datetime(845510400000000000),
            pd.to_datetime(-145497600000000000)
        ])
        expect_reverse_call_count = 0

        pd.testing.assert_series_equal(result, expect)
        self.assertEqual(
            transformer.null_transformer.reverse_transform.call_count,
            expect_reverse_call_count,
            "NullTransformer.reverse_transform won't be called when nan is ignore"
        )
