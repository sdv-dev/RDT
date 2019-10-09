from unittest import TestCase
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from rdt.transformers import NullTransformer


class TestNullTransformer(TestCase):

    def test___init__(self):
        """Test default instance"""
        # Run
        transformer = NullTransformer(None)

        # Asserts
        self.assertIsNone(transformer.null_column, "null_column is None by default")
        self.assertFalse(transformer.copy, "copy is False by default")

    def test_fit_null_column_none_true(self):
        """Test fit with null_columns equal to None, None values in data"""
        # Setup
        data = pd.Series([1.5, None, 2.5])

        # Run
        fill_value = None
        null_column = None

        transformer = NullTransformer(fill_value, null_column=null_column)
        transformer.fit(data)

        # Asserts
        self.assertTrue(
            transformer._null_column,
            "_null_column must be true when there are None values as input"
        )

    def test_fit_null_column_none_false(self):
        """Test fit with null_columns equal to None, no None values in data"""
        # Setup
        data = pd.Series([1.5, 3.0, 2.5])

        # Run
        fill_value = None
        null_column = None

        transformer = NullTransformer(fill_value, null_column=null_column)
        transformer.fit(data)

        # Asserts
        self.assertFalse(
            transformer._null_column,
            "_null_column must be true when there are None values as input"
        )

    def test_fit_null_column_false(self):
        """Test fit with null_column equal to False"""
        # Setup
        data = pd.Series([1.5, None, 2.5])

        # Run
        fill_value = None
        null_column = False

        transformer = NullTransformer(fill_value, null_column=null_column)
        transformer.fit(data)

        # Asserts
        self.assertFalse(
            transformer._null_column,
            "_null_column must be False when null_column is False"
        )

    def test_fit_null_column_true(self):
        """Test fit with null_column equal to True"""
        # Setup
        data = pd.Series([1.5, None, 2.5])

        # Run
        fill_value = None
        null_column = True

        transformer = NullTransformer(fill_value, null_column=null_column)
        transformer.fit(data)

        # Asserts
        self.assertTrue(
            transformer._null_column,
            "_null_column must be True when null_column is True"
        )

    def test_transform_without_nulls(self):
        """Test transform data with nulls equal to False"""
        # Setup
        data = pd.Series([1.5, None, 2.5])

        # Run
        transformer = Mock()
        transformer.nulls = False

        result = NullTransformer.transform(transformer, data)

        # Asserts
        expect = np.array([1.5, np.nan, 2.5])

        np.testing.assert_array_equal(result, expect, "Unexpected transformed data")

    @pytest.mark.filterwarnings("ignore")
    def test_transform_with_nulls_fillvalue_no_copy(self):
        """Test transform data with nulls equal to True and fill_value

        NOTE: This test will execute a code that raise a warning.
              We add the filterwarning to mute it since it's not a real warning in the test.
        """
        # Setup
        data = pd.Series([1.5, None, 2.5])

        # Run
        transformer = Mock()
        transformer.nulls = True
        transformer.copy = False
        transformer.fill_value = 0
        transformer._null_column = False

        result = NullTransformer.transform(transformer, data)

        # Asserts
        expect = np.array([1.5, 0, 2.5])

        np.testing.assert_array_equal(result, expect, "Unexpected transformed data")

    @pytest.mark.filterwarnings("ignore")
    def test_transform_with_nulls_fillvalue_copy(self):
        """Test transform data with nulls equal to True and fill_value"""
        # Setup
        data = pd.Series([1.5, None, 2.5])

        # Run
        transformer = Mock()
        transformer.nulls = True
        transformer.copy = True
        transformer.fill_value = 0
        transformer._null_column = False

        result = NullTransformer.transform(transformer, data)

        # Asserts
        expect = np.array([1.5, 0, 2.5])

        np.testing.assert_array_equal(result, expect, "Unexpected transformed data")

    def test_transform_with_nulls_null_column(self):
        """Test transform data with nulls equal to True and null_column"""
        # Setup
        data = pd.Series([1.5, None, 2.5])

        # Run
        transformer = Mock()
        transformer.nulls = True
        transformer.fill_value = None
        transformer._null_column = True

        result = NullTransformer.transform(transformer, data)

        # Asserts
        expect = np.array([[1.5, 0], [np.nan, 1], [2.5, 0]])

        np.testing.assert_array_equal(result, expect, "Unexpected transformed data")

    def test_reverse_transform_no_nulls(self):
        """Test reverse_transform with no nulls"""
        # Setup
        data = pd.Series([1.5, None, 2.5])

        # Run
        transformer = Mock()
        transformer.nulls = False

        result = NullTransformer.reverse_transform(transformer, data)

        # Asserts
        expect = pd.Series([1.5, None, 2.5])

        np.testing.assert_array_equal(result, expect, "Unextected reverse data")

    def test_reverse_transform_nulls_and_null_column(self):
        """Test reverse_transform with nulls and null_column"""
        # Setup
        data = np.array([[1.5, 0], [0.6, 1], [2.5, 0]])

        # Run
        transformer = Mock()
        transformer.nulls = True
        transformer._null_column = True

        result = NullTransformer.reverse_transform(transformer, data)

        # Asserts
        expect = pd.Series([1.5, np.nan, 2.5])

        np.testing.assert_array_equal(result, expect, "Unextected reverse data")

    def test_reverse_transform_nulls_and_not_null_column(self):
        """Test reverse_transform with nulls and not null_column"""
        # Setup
        data = np.array([1.5, 5.0, 2.5])

        # Run
        transformer = Mock()
        transformer.nulls = True
        transformer._null_column = False
        transformer.fill_value = 5.0

        result = NullTransformer.reverse_transform(transformer, data)

        # Asserts
        expect = pd.Series([1.5, np.nan, 2.5])

        np.testing.assert_array_equal(result, expect, "Unextected reverse data")
