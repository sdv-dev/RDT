import unittest

import numpy as np
import pandas as pd

from rdt.transformers.NumberTransformer import NumberTransformer


class TestNumberTransformer(unittest.TestCase):

    def test___init__(self):
        """On init, sets type to number"""

        # Run
        transformer = NumberTransformer()

        # Check
        assert transformer.type == 'number'

    def test_fit_transform(self):
        """fit_transform sets internal state and transforms data."""
        # Setup
        col = pd.DataFrame({
            'age': [62, 27, 5, 34, 62]
        })

        col_meta = {
            "name": "age",
            "type": "number",
            "subtype": "integer",
        }
        transformer = NumberTransformer()
        expected_result = pd.DataFrame({'age': [62, 27, 5, 34, 62]})

        # Run
        result = transformer.fit_transform(col, col_meta, False)

        # Check
        assert result.equals(expected_result)

    def test_fit_transform_missing(self):
        """Sets internal state and transforms data with missing values."""
        # Setup
        col = pd.DataFrame({
            'age': [62, 27, np.nan, 34, 62],
        })
        col_meta = {
            "name": "age",
            "type": "number",
            "subtype": "integer",
        }
        transformer = NumberTransformer()
        expected_result = pd.DataFrame({
            'age': [62, 27, 46, 34, 62],
            '?age': [1, 1, 0, 1, 1]
        })

        # Run
        result = transformer.fit_transform(col, col_meta, True)

        # Check
        assert result.equals(expected_result)

    def test_reverse_transform(self):
        """Checks the conversion of the data back into original format."""
        # Setup
        col = pd.DataFrame({
            'age': [34, 23, 27, 31, 39]
        })
        col_meta = {
            'name': 'age',
            'subtype': 'integer',
            'type': 'number'
        }
        transformer = NumberTransformer()
        expected_result = pd.DataFrame({'age': [34, 23, 27, 31, 39]})

        # Run
        result = transformer.reverse_transform(col, col_meta, False)

        # Check
        assert result.equals(expected_result)

    def test_reverse_transform_nan(self):
        """Checks that nans are handled correctly in reverse transformation."""
        # Setup
        col = pd.DataFrame({
            'age': [34, 23, 27, 31, 39]
        })
        col_meta = {
            'name': 'age',
            'subtype': 'integer',
            'type': 'number'
        }
        transformer = NumberTransformer()
        transformer.fit_transform(col, col_meta, False)

        col2 = pd.DataFrame({
            'age': [0, 10, 20, 30, np.nan]
        })

        expected_result = pd.DataFrame({
            'age': [0, 10, 20, 30, transformer.default_val]
        })

        # Run
        result = transformer.reverse_transform(col2, col_meta, False)

        # Check
        assert result.equals(expected_result)

    def test_reverse_transform_missing(self):
        """Sets internal state and transforms data with missing values."""
        # Setup
        col = pd.DataFrame({
            'age': [34, 23, 0, 31, 39],
            '?age': [1, 1, 0, 1, 1]
        })
        col_meta = {
            'name': 'age',
            'subtype': 'integer',
            'type': 'number'
        }
        transformer = NumberTransformer()
        expected_result = pd.DataFrame({'age': [34, 23, np.nan, 31, 39]})

        # Run
        result = transformer.reverse_transform(col, col_meta, True)

        # Check
        assert result.equals(expected_result)

    def test_get_val_subtype_integer(self):
        """Rounds off the value and returns it."""

        # Setup
        col = pd.DataFrame({
            'age': [62, 35, 24]
        })

        transformer = NumberTransformer()
        transformer.col_name = 'age'
        transformer.subtype = 'integer'
        transformer.default_val = col.age[0]
        expected_result = pd.Series([62, 35, 24])

        # Run
        result = col.apply(transformer.get_val, axis=1)

        # Check
        assert result.equals(expected_result)

    def test_get_val_subtype_not_integer(self):
        """If subtype is not integer and there are not null values, returns the same value."""

        # Setup
        col = pd.DataFrame({
            'age': [62.5, 35.5, 24.3]
        })

        transformer = NumberTransformer()
        transformer.col_name = 'age'
        transformer.subtype = 'decimal'
        transformer.default_val = col.age[0]
        expected_result = pd.Series([62.5, 35.5, 24.3])

        # Run
        result = col.apply(transformer.get_val, axis=1)

        # Check
        assert result.equals(expected_result)

    def test_get_val_null_value(self):
        """get_val return the default value for null values if subtype is not integer."""

        # Setup
        col = pd.DataFrame({
            'age': [4, np.nan, np.nan]
        })

        transformer = NumberTransformer()
        transformer.col_name = 'age'
        transformer.subtype = 'decimal'
        transformer.default_val = 5.0
        expected_result = pd.Series([4.0, 5.0, 5.0])
        # Run
        result = col.apply(transformer.get_val, axis=1)

        # Check
        assert result.equals(expected_result)

    def test_get_val_error(self):
        """Returns 'default_val' when there is a ValueError."""
        # Setup
        col = pd.DataFrame({
            'age': [4, 'hoo', 13]
        })

        transformer = NumberTransformer()
        transformer.col_name = 'age'
        transformer.subtype = 'integer'
        transformer.default_val = 999
        expected_result = pd.Series([4, 999, 13])

        # Run
        result = col.apply(transformer.get_val, axis=1)

        # Check
        assert result.equals(expected_result)

    def test_get_number_converter(self):
        """If meta 'integer', cast values to int."""
        # Setup
        transformer = NumberTransformer()
        col_name = 'age'
        converter = transformer.get_number_converter(col_name, 'integer')
        data = pd.DataFrame({
            'age': [0.5, 10.1, 3]
        })
        expected_result = pd.Series([0, 10, 3], name='age')

        # Run
        result = data.apply(converter, axis=1)

        # Check
        assert result.equals(expected_result)
