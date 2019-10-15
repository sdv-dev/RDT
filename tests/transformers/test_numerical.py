from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from rdt.transformers import NumericalTransformer


class TestNumericalTransformer(TestCase):

    def test___init__(self):
        """Test default instance"""
        # Run
        transformer = NumericalTransformer()

        # Asserts
        self.assertEqual(transformer.nan, 'mean', "Unexpected nan")
        self.assertIsNone(transformer.null_column, "null_column is None by default")
        self.assertIsNone(transformer.dtype, "dtype is None by default")

    def test_fit_nan_mean_array(self):
        """Test fit nan mean with numpy.array"""
        # Setup
        data = np.array([1.5, None, 2.5])

        # Run
        transformer = NumericalTransformer(dtype=np.float, nan='mean')
        transformer.fit(data)

        # Asserts
        expect_fill_value = 2.0
        expect_dtype = np.float

        self.assertEqual(
            transformer.null_transformer.fill_value,
            expect_fill_value,
            "Data mean is wrong"
        )

        self.assertEqual(
            transformer._dtype,
            expect_dtype,
            "Expected dtype: float"
        )

    def test_fit_nan_mean_series(self):
        """Test fit nan mean with pandas.Series"""
        # Setup
        data = pd.Series([1.5, None, 2.5])

        # Run
        transformer = NumericalTransformer(dtype=np.float, nan='mean')
        transformer.fit(data)

        # Asserts
        expect_fill_value = 2.0
        expect_dtype = np.float

        self.assertEqual(
            transformer.null_transformer.fill_value,
            expect_fill_value,
            "Data mean is wrong"
        )

        self.assertEqual(
            transformer._dtype,
            expect_dtype,
            "Expected dtype: float"
        )

    def test_fit_nan_mode_array(self):
        """Test fit nan mode with numpy.array"""
        # Setup
        data = np.array([1.5, None, 2.5])

        # Run
        transformer = NumericalTransformer(dtype=np.float, nan='mode')
        transformer.fit(data)

        # Asserts
        expect_fill_value = 1.5
        expect_dtype = np.float

        self.assertEqual(
            transformer.null_transformer.fill_value,
            expect_fill_value,
            "Data mean is wrong"
        )

        self.assertEqual(
            transformer._dtype,
            expect_dtype,
            "Expected dtype: float"
        )

    def test_fit_nan_mode_series(self):
        """Test fit nan mode with pandas.Series"""
        # Setup
        data = pd.Series([1.5, None, 2.5])

        # Run
        transformer = NumericalTransformer(dtype=np.float, nan='mode')
        transformer.fit(data)

        # Asserts
        expect_fill_value = 1.5
        expect_dtype = np.float

        self.assertEqual(
            transformer.null_transformer.fill_value,
            expect_fill_value,
            "Data mean is wrong"
        )

        self.assertEqual(
            transformer._dtype,
            expect_dtype,
            "Expected dtype: float"
        )

    def test_fit_nan_ignore_array(self):
        """Test fit nan ignore with numpy.array"""
        # Setup
        data = np.array([1.5, None, 2.5])

        # Run
        transformer = NumericalTransformer(dtype=np.float, nan=None)
        transformer.fit(data)

        # Asserts
        expect_fill_value = None
        expect_dtype = np.float

        self.assertEqual(
            transformer.null_transformer.fill_value,
            expect_fill_value,
            "Data mean is wrong"
        )

        self.assertEqual(
            transformer._dtype,
            expect_dtype,
            "Expected dtype: float"
        )

    def test_fit_nan_ignore_series(self):
        """Test fit nan ignore with pandas.Series"""
        # Setup
        data = pd.Series([1.5, None, 2.5])

        # Run
        transformer = NumericalTransformer(dtype=np.float, nan=None)
        transformer.fit(data)

        # Asserts
        expect_fill_value = None
        expect_dtype = np.float

        self.assertEqual(
            transformer.null_transformer.fill_value,
            expect_fill_value,
            "Data mean is wrong"
        )

        self.assertEqual(
            transformer._dtype,
            expect_dtype,
            "Expected dtype: float"
        )

    def test_fit_nan_other_array(self):
        """Test fit nan custom value with numpy.array"""
        # Setup
        data = np.array([1.5, None, 2.5])

        # Run
        transformer = NumericalTransformer(dtype=np.float, nan=0)
        transformer.fit(data)

        # Asserts
        expect_fill_value = 0
        expect_dtype = np.float

        self.assertEqual(
            transformer.null_transformer.fill_value,
            expect_fill_value,
            "Data mean is wrong"
        )

        self.assertEqual(
            transformer._dtype,
            expect_dtype,
            "Expected dtype: float"
        )

    def test_fit_nan_other_series(self):
        """Test fit nan custom value with pandas.Series"""
        # Setup
        data = pd.Series([1.5, None, 2.5])

        # Run
        transformer = NumericalTransformer(dtype=np.float, nan=0)
        transformer.fit(data)

        # Asserts
        expect_fill_value = 0
        expect_dtype = np.float

        self.assertEqual(
            transformer.null_transformer.fill_value,
            expect_fill_value,
            "Data mean is wrong"
        )

        self.assertEqual(
            transformer._dtype,
            expect_dtype,
            "Expected dtype: float"
        )

    def test_transform_array(self):
        """Test transform numpy.array"""
        # Setup
        data = np.array([1.5, None, 2.5])

        # Run
        transformer = Mock()
        NumericalTransformer.transform(transformer, data)

        # Asserts
        expect_call_count = 1

        self.assertEqual(
            transformer.null_transformer.transform.call_count,
            expect_call_count,
            "Transform must be called only once"
        )

    def test_transform_series(self):
        """Test transform pandas.Series"""
        # Setup
        data = pd.Series([1.5, None, 2.5])

        # Run
        transformer = Mock()
        NumericalTransformer.transform(transformer, data)

        # Asserts
        expect_call_count = 1

        self.assertEqual(
            transformer.null_transformer.transform.call_count,
            expect_call_count,
            "Transform must be called only once"
        )

    def test_reverse_transform_nan_ignore(self):
        """Test reverse_transform with nan equal to ignore"""
        # Setup
        data = pd.Series([1.5, None, 2.5])

        # Run
        transformer = Mock()
        transformer.nan = None
        transformer._dtype = np.float

        result = NumericalTransformer.reverse_transform(transformer, data)

        # Asserts
        expect = pd.Series([1.5, None, 2.5])
        expected_reverse_transform_call_count = 0

        pd.testing.assert_series_equal(result, expect)
        self.assertEqual(
            transformer.null_transformer.reverse_transform.call_count,
            expected_reverse_transform_call_count,
            "NullTransformer.reverse_transform can't be called when nan is ignore"
        )

    def test_reverse_transform_nan_not_ignore(self):
        """Test reverse_transform with nan not equal to ignore"""
        # Setup
        data = pd.Series([1.5, 2.0, 2.5])
        reversed_data = pd.Series([1.5, 2.0, 2.5])

        # Run
        transformer = Mock()
        transformer.nan = 'mean'
        transformer._dtype = np.float
        transformer.null_transformer.nulls = False
        transformer.null_transformer.reverse_transform.return_value = reversed_data

        NumericalTransformer.reverse_transform(transformer, data)

        # Asserts
        expected_reverse_transform_call_count = 1

        self.assertEqual(
            transformer.null_transformer.reverse_transform.call_count,
            expected_reverse_transform_call_count,
            "NullTransformer.reverse_transform must be called at least once"
        )

    @patch('numpy.round')
    def test_reverse_transform_dtype_int(self, numpy_mock):
        """Test reverse_transform with dtype equal to int"""
        # Setup
        numpy_mock.return_value = pd.Series([3, 2, 3])
        data = pd.Series([3.0, 2.0, 3.0])

        # Run
        transformer = Mock()
        transformer.nan = None
        transformer._dtype = np.int

        result = NumericalTransformer.reverse_transform(transformer, data)

        # Asserts
        expect = pd.Series([3.0, 2.0, 3.0])
        expected_reverse_transform_call_count = 0

        pd.testing.assert_series_equal(result, expect)
        self.assertEqual(
            transformer.null_transformer.reverse_transform.call_count,
            expected_reverse_transform_call_count,
            "NullTransformer.reverse_transform must be called at least once"
        )
