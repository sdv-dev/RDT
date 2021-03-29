from unittest import TestCase
from unittest.mock import Mock

import numpy as np
import pandas as pd

from rdt.transformers import BooleanTransformer


class TestBooleanTransformer(TestCase):

    def test___init__(self):
        """Test default instance"""
        # Run
        transformer = BooleanTransformer()

        # Asserts
        self.assertEqual(transformer.nan, -1, "Unexpected nan")
        self.assertIsNone(transformer.null_column, "null_column is None by default")

    def test_fit_nan_ignore(self):
        """Test fit nan equal to ignore"""
        # Setup
        data = pd.Series([False, True, True, False, True])

        # Run
        transformer = BooleanTransformer(nan=None)
        transformer.fit(data)

        # Asserts
        expect_fill_value = None

        self.assertEqual(
            transformer.null_transformer.fill_value,
            expect_fill_value,
            "Unexpected fill value"
        )

    def test_fit_nan_not_ignore(self):
        """Test fit nan not equal to ignore"""
        # Setup
        data = pd.Series([False, True, True, False, True])

        # Run
        transformer = BooleanTransformer(nan=0)
        transformer.fit(data)

        # Asserts
        expect_fill_value = 0

        self.assertEqual(
            transformer.null_transformer.fill_value,
            expect_fill_value,
            "Unexpected fill value"
        )

    def test_fit_array(self):
        """Test fit with numpy.array"""
        # Setup
        data = np.array([False, True, True, False, True])

        # Run
        transformer = BooleanTransformer(nan=0)
        transformer.fit(data)

        # Asserts
        expect_fill_value = 0

        self.assertEqual(
            transformer.null_transformer.fill_value,
            expect_fill_value,
            "Unexpected fill value"
        )

    def test_transform_series(self):
        """Test transform pandas.Series"""
        # Setup
        data = pd.Series([False, True, None, True, False])

        # Run
        transformer = Mock()

        BooleanTransformer.transform(transformer, data)

        # Asserts
        expect_call_count = 1
        expect_call_args = pd.Series([0, 1, None, 1, 0], dtype=object)

        self.assertEqual(
            transformer.null_transformer.transform.call_count,
            expect_call_count,
            "NullTransformer.transform must be called one time"
        )
        pd.testing.assert_series_equal(
            transformer.null_transformer.transform.call_args[0][0],
            expect_call_args
        )

    def test_transform_array(self):
        """Test transform numpy.array"""
        # Setup
        data = np.array([False, True, None, True, False])

        # Run
        transformer = Mock()

        BooleanTransformer.transform(transformer, data)

        # Asserts
        expect_call_count = 1
        expect_call_args = pd.Series([0, 1, None, 1, 0], dtype=object)

        self.assertEqual(
            transformer.null_transformer.transform.call_count,
            expect_call_count,
            "NullTransformer.transform must be called one time"
        )
        pd.testing.assert_series_equal(
            transformer.null_transformer.transform.call_args[0][0],
            expect_call_args
        )

    def test_reverse_transform_nan_ignore(self):
        """Test reverse_transform with nan equal to ignore"""
        # Setup
        data = np.array([0.0, 1.0, 0.0, 1.0, 0.0])

        # Run
        transformer = Mock()
        transformer.nan = None

        result = BooleanTransformer.reverse_transform(transformer, data)

        # Asserts
        expect = np.array([False, True, False, True, False])
        expect_call_count = 0

        np.testing.assert_equal(result, expect)
        self.assertEqual(
            transformer.null_transformer.reverse_transform.call_count,
            expect_call_count,
            "NullTransformer.reverse_transform should not be called when nan is ignore"
        )

    def test_reverse_transform_nan_not_ignore(self):
        """Test reverse_transform with nan not equal to ignore"""
        # Setup
        data = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
        transformed_data = np.array([0.0, 1.0, 0.0, 1.0, 0.0])

        # Run
        transformer = Mock()
        transformer.nan = 0
        transformer.null_transformer.reverse_transform.return_value = transformed_data

        result = BooleanTransformer.reverse_transform(transformer, data)

        # Asserts
        expect = np.array([False, True, False, True, False])
        expect_call_count = 1

        np.testing.assert_equal(result, expect)
        self.assertEqual(
            transformer.null_transformer.reverse_transform.call_count,
            expect_call_count,
            "NullTransformer.reverse_transform should not be called when nan is ignore"
        )

    def test_reverse_transform_not_null_values(self):
        """Test reverse_transform not null values correctly"""
        # Setup
        data = np.array([1., 0., 1.])

        # Run
        transformer = Mock()
        transformer.nan = None

        result = BooleanTransformer.reverse_transform(transformer, data)

        # Asserts
        expected = np.array([True, False, True])

        assert isinstance(result, pd.Series)
        np.testing.assert_equal(result.values, expected)

    def test_reverse_transform_2d_ndarray(self):
        """Test reverse_transform not null values correctly"""
        # Setup
        data = np.array([[1.], [0.], [1.]])

        # Run
        transformer = Mock()
        transformer.nan = None

        result = BooleanTransformer.reverse_transform(transformer, data)

        # Asserts
        expected = np.array([True, False, True])

        assert isinstance(result, pd.Series)
        np.testing.assert_equal(result.values, expected)
