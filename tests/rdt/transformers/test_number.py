import unittest

import numpy as np
import pandas as pd

from rdt.transformers.number import NumberTransformer


class TestNumberTransformer(unittest.TestCase):

    def test___init__(self):
        """On init, sets type to number"""
        # Setup
        column_metadata = {
            "name": "age",
            "type": "number",
            "subtype": "integer",
        }
        # Run
        transformer = NumberTransformer(column_metadata)

        # Check
        assert transformer.type == 'number'
        assert transformer.subtype == 'integer'
        assert transformer.col_name == 'age'

    def test_fit_transform(self):
        """fit_transform sets internal state and transforms data."""
        # Setup
        col = pd.DataFrame({
            'age': [62, 27, 5, 34, 62]
        })

        column_metadata = {
            "name": "age",
            "type": "number",
            "subtype": "integer",
        }
        transformer = NumberTransformer(column_metadata)
        expected_result = pd.DataFrame({'age': [62, 27, 5, 34, 62]})

        # Run
        result = transformer.fit_transform(col)

        # Check
        assert result.equals(expected_result)

    def test_reverse_transform(self):
        """Checks the conversion of the data back into original format."""
        # Setup
        col = pd.DataFrame({
            'age': [34, 23, 27, 31, 39]
        })
        column_metadata = {
            'name': 'age',
            'subtype': 'integer',
            'type': 'number'
        }
        transformer = NumberTransformer(column_metadata)
        expected_result = pd.DataFrame({'age': [34, 23, 27, 31, 39]})

        # Run
        result = transformer.reverse_transform(col)

        # Check
        assert result.equals(expected_result)

    def test_reverse_transform_nan(self):
        """Checks that nans are handled correctly in reverse transformation."""
        # Setup
        col = pd.DataFrame({
            'age': [34, 23, 27, 31, 39]
        })
        column_metadata = {
            'name': 'age',
            'subtype': 'integer',
            'type': 'number'
        }
        transformer = NumberTransformer(column_metadata)
        transformer.fit_transform(col)

        col2 = pd.DataFrame({
            'age': [0, 10, 20, 30, np.nan]
        })

        expected_result = pd.DataFrame({
            'age': [0, 10, 20, 30, transformer.default_val]
        })

        # Run
        result = transformer.reverse_transform(col2)

        # Check
        assert result.equals(expected_result)

    def test_get_val_subtype_integer(self):
        """Rounds off the value and returns it."""

        # Setup
        col = pd.DataFrame({
            'age': [62, 35, 24]
        })
        column_metadata = {
            'name': 'age',
            'subtype': 'integer',
            'type': 'number'
        }

        transformer = NumberTransformer(column_metadata)
        transformer.fit(col)
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
        column_metadata = {
            'name': 'age',
            'subtype': 'decimal',
            'type': 'number'
        }

        transformer = NumberTransformer(column_metadata)
        transformer.fit(col)
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
        column_metadata = {
            'name': 'age',
            'subtype': 'decimal',
            'type': 'number'
        }

        transformer = NumberTransformer(column_metadata)
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
        column_metadata = {
            'name': 'age',
            'subtype': 'integer',
            'type': 'number'
        }

        transformer = NumberTransformer(column_metadata)
        transformer.default_val = 999
        expected_result = pd.Series([4, 999, 13])

        # Run
        result = col.apply(transformer.get_val, axis=1)

        # Check
        assert result.equals(expected_result)

    def test_safe_round(self):
        """If meta 'integer', cast values to int."""
        # Setup
        column_metadata = {
            'name': 'age',
            'subtype': 'integer',
            'type': 'number'
        }
        transformer = NumberTransformer(column_metadata)
        data = pd.DataFrame({
            'age': [0.5, 10.1, 3]
        })
        expected_result = pd.Series([0, 10, 3], name='age')

        # Run
        result = data.apply(transformer.safe_round, axis=1)

        # Check
        assert result.equals(expected_result)
