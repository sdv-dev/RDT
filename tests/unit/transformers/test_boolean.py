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
        assert transformer.nan == -1
        assert transformer.null_column is None

    def test__fit_nan_ignore(self):
        """Test _fit nan equal to ignore"""
        # Setup
        data = pd.Series([False, True, True, False, True])

        # Run
        transformer = BooleanTransformer(nan=None)
        transformer._fit(data)

        # Asserts
        assert transformer.null_transformer.fill_value is None

    def test__fit_nan_not_ignore(self):
        """Test _fit nan not equal to ignore"""
        # Setup
        data = pd.Series([False, True, True, False, True])

        # Run
        transformer = BooleanTransformer(nan=0)
        transformer._fit(data)

        # Asserts
        assert transformer.null_transformer.fill_value == 0

    def test__fit_array(self):
        """Test _fit with numpy.array"""
        # Setup
        data = np.array([False, True, True, False, True])

        # Run
        transformer = BooleanTransformer(nan=0)
        transformer._fit(data)

        # Asserts
        assert transformer.null_transformer.fill_value == 0

    def test__transform_series(self):
        """Test transform pandas.Series"""
        # Setup
        data = pd.Series([False, True, None, True, False])

        # Run
        transformer = Mock()

        BooleanTransformer._transform(transformer, data)

        # Asserts
        expect_call_count = 1
        expect_call_args = pd.Series([0., 1., None, 1., 0.], dtype=float)

        assert transformer.null_transformer.transform.call_count == expect_call_count
        pd.testing.assert_series_equal(
            transformer.null_transformer.transform.call_args[0][0],
            expect_call_args
        )

    def test__transform_array(self):
        """Test transform numpy.array"""
        # Setup
        data = np.array([False, True, None, True, False])

        # Run
        transformer = Mock()

        BooleanTransformer._transform(transformer, data)

        # Asserts
        expect_call_count = 1
        expect_call_args = pd.Series([0., 1., None, 1., 0.], dtype=float)

        assert transformer.null_transformer.transform.call_count == expect_call_count
        pd.testing.assert_series_equal(
            transformer.null_transformer.transform.call_args[0][0],
            expect_call_args
        )

    def test__reverse_transform_nan_ignore(self):
        """Test _reverse_transform with nan equal to ignore"""
        # Setup
        data = np.array([0.0, 1.0, 0.0, 1.0, 0.0])

        # Run
        transformer = Mock()
        transformer.nan = None

        result = BooleanTransformer._reverse_transform(transformer, data)

        # Asserts
        expect = np.array([False, True, False, True, False])
        expect_call_count = 0

        np.testing.assert_equal(result, expect)
        assert transformer.null_transformer.reverse_transform.call_count == expect_call_count

    def test__reverse_transform_nan_not_ignore(self):
        """Test _reverse_transform with nan not equal to ignore"""
        # Setup
        data = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
        transformed_data = np.array([0.0, 1.0, 0.0, 1.0, 0.0])

        # Run
        transformer = Mock()
        transformer.nan = 0
        transformer.null_transformer.reverse_transform.return_value = transformed_data

        result = BooleanTransformer._reverse_transform(transformer, data)

        # Asserts
        expect = np.array([False, True, False, True, False])
        expect_call_count = 1

        np.testing.assert_equal(result, expect)
        assert transformer.null_transformer.reverse_transform.call_count == expect_call_count

    def test__reverse_transform_not_null_values(self):
        """Test _reverse_transform not null values correctly"""
        # Setup
        data = np.array([1., 0., 1.])

        # Run
        transformer = Mock()
        transformer.nan = None

        result = BooleanTransformer._reverse_transform(transformer, data)

        # Asserts
        expected = np.array([True, False, True])

        assert isinstance(result, pd.Series)
        np.testing.assert_equal(result.values, expected)

    def test__reverse_transform_2d_ndarray(self):
        """Test _reverse_transform not null values correctly"""
        # Setup
        data = np.array([[1.], [0.], [1.]])

        # Run
        transformer = Mock()
        transformer.nan = None

        result = BooleanTransformer._reverse_transform(transformer, data)

        # Asserts
        expected = np.array([True, False, True])

        assert isinstance(result, pd.Series)
        np.testing.assert_equal(result.values, expected)

    def test__reverse_transform_float_values(self):
        """Test the ``_reverse_transform`` method with decimals.

        Expect that the ``_reverse_transform`` method handles decimal inputs
        correctly by rounding them.

        Input:
            - Transformed data with decimal values.
        Output:
            - Reversed transformed data.
        """
        # Setup
        data = np.array([1.2, 0.32, 1.01])
        transformer = Mock()
        transformer.nan = None

        # Run
        result = BooleanTransformer._reverse_transform(transformer, data)

        # Asserts
        expected = np.array([True, False, True])

        assert isinstance(result, pd.Series)
        np.testing.assert_equal(result.values, expected)

    def test__reverse_transform_float_values_out_of_range(self):
        """Test the ``_reverse_transform`` method with decimals that are out of range.

        Expect that the ``_reverse_transform`` method handles decimal inputs
        correctly by rounding them. If the rounded decimal inputs are < 0 or > 1, expect
        expect them to be clipped.

        Input:
            - Transformed data with decimal values, some of which round to < 0 or > 1.
        Output:
            - Reversed transformed data.
        """
        # Setup
        data = np.array([1.9, -0.7, 1.01])
        transformer = Mock()
        transformer.nan = None

        # Run
        result = BooleanTransformer._reverse_transform(transformer, data)

        # Asserts
        expected = np.array([True, False, True])

        assert isinstance(result, pd.Series)
        np.testing.assert_equal(result.values, expected)
