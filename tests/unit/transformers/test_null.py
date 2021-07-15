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

    def test_fit_fill_value_mean(self):
        """If fill_value is mean, _fill_value is the mean of the input data."""
        # Setup
        data = pd.Series([1, 2, 3])

        # Run
        transformer = NullTransformer(fill_value='mean')
        transformer.fit(data)

        # Asserts
        assert transformer._fill_value == 2

    def test_fit_fill_value_mean_nulls(self):
        """If fill_value is mean, _fill_value is the mean of the input data."""
        # Setup
        data = pd.Series([1, None, 3])

        # Run
        transformer = NullTransformer(fill_value='mean')
        transformer.fit(data)

        # Asserts
        assert transformer._fill_value == 2

    def test_fit_fill_value_mean_all_nulls(self):
        """If fill_value is mean and all the values are null, _fill_value is 0."""
        # Setup
        data = pd.Series([None, None, None])

        # Run
        transformer = NullTransformer(fill_value='mean')
        transformer.fit(data)

        # Asserts
        assert transformer._fill_value == 0

    def test_fit_fill_value_mode(self):
        """If fill_value is mode, _fill_value is the mode of the input data."""
        # Setup
        data = pd.Series([1, 2, 3, 3, 4])

        # Run
        transformer = NullTransformer(fill_value='mode')
        transformer.fit(data)

        # Asserts
        assert transformer._fill_value == 3

    def test_fit_fill_value_mode_nulls(self):
        """If fill_value is mode, _fill_value is the mode of the input data."""
        # Setup
        data = pd.Series([1, None, 3, 3, 4])

        # Run
        transformer = NullTransformer(fill_value='mode')
        transformer.fit(data)

        # Asserts
        assert transformer._fill_value == 3

    def test_fit_fill_value_mode_all_nulls(self):
        """If fill_value is mode and all the values are null, _fill_value is 0."""
        # Setup
        data = pd.Series([None, None, None])

        # Run
        transformer = NullTransformer(fill_value='mode')
        transformer.fit(data)

        # Asserts
        assert transformer._fill_value == 0

    def test_fit_fill_value(self):
        """If fill_value is something else, _fill_value is fill_value."""
        # Setup
        data = pd.Series([1, 2, 3, 3, 4])

        # Run
        transformer = NullTransformer(fill_value=43)
        transformer.fit(data)

        # Asserts
        assert transformer._fill_value == 43

    def test_fit_fill_value_all_nulls(self):
        """If fill_value is something else, _fill_value is fill_value."""
        # Setup
        data = pd.Series([None, None, None])

        # Run
        transformer = NullTransformer(fill_value=43)
        transformer.fit(data)

        # Asserts
        assert transformer._fill_value == 43

    def test_fit_null_column_true(self):
        """If null_column is true, _null_column is always true."""
        # Setup
        nonulls = pd.Series([1, 2, 3])
        nulls = pd.Series([1, None, 3])

        # Run
        nonulls_transformer = NullTransformer(fill_value=0, null_column=True)
        nonulls_transformer.fit(nonulls)
        nulls_transformer = NullTransformer(fill_value=0, null_column=True)
        nulls_transformer.fit(nulls)

        # Asserts
        assert nonulls_transformer._null_column
        assert nulls_transformer._null_column

    def test_fit_null_column_false(self):
        """If null_column is false, _null_column is always false."""
        # Setup
        nonulls = pd.Series([1, 2, 3])
        nulls = pd.Series([1, None, 3])

        # Run
        nonulls_transformer = NullTransformer(fill_value=0, null_column=False)
        nonulls_transformer.fit(nonulls)
        nulls_transformer = NullTransformer(fill_value=0, null_column=False)
        nulls_transformer.fit(nulls)

        # Asserts
        assert not nonulls_transformer._null_column
        assert not nulls_transformer._null_column

    def test_fit_null_column_none(self):
        """If null_column is None, _null_column is true only if there are Nulls."""
        # Setup
        nonulls = pd.Series([1, 2, 3])
        nulls = pd.Series([1, None, 3])

        # Run
        nonulls_transformer = NullTransformer(fill_value=0, null_column=None)
        nonulls_transformer.fit(nonulls)
        nulls_transformer = NullTransformer(fill_value=0, null_column=None)
        nulls_transformer.fit(nulls)

        # Asserts
        assert not nonulls_transformer._null_column
        assert nulls_transformer._null_column

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
        transformer._fill_value = 0
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
        transformer._fill_value = 0
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
        transformer._fill_value = None
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
        transformer._fill_value = 5.0

        result = NullTransformer.reverse_transform(transformer, data)

        # Asserts
        expect = pd.Series([1.5, np.nan, 2.5])

        np.testing.assert_array_equal(result, expect, "Unextected reverse data")
