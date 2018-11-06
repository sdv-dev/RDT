import unittest

import numpy as np
import pandas as pd

from rdt.transformers.NullTransformer import NullTransformer


class TestNullTransformer(unittest.TestCase):
    def test___init__(self):
        """On __init__ set type to number and datetime."""

        # Run
        transformer = NullTransformer()

        # Check
        assert transformer.type == ['datetime', 'number']

    def test_fit_transform_isnull(self):
        """It will replace nan values with 0 and creats a new column."""

        # Setup
        col = pd.Series([62, np.nan, np.nan, np.nan, np.nan], name='age')
        col_meta = {
            'name': 'age',
            'type': 'number'
        }
        transformer = NullTransformer()

        expected_result = pd.DataFrame({
            'age': [62.0, 62.0, 62.0, 62.0, 62.0],
            '?age': [1, 0, 0, 0, 0]
        })

        # Run
        result = transformer.fit_transform(col, col_meta)

        # Check
        assert result.equals(expected_result)

    def test_fit_transform_notnull(self):
        """Creates a new column with the mean of the values."""

        # Setup
        col = pd.Series([62, 53, 53, 45, np.nan])
        col_meta = {
            'name': 'age',
            'type': 'number'
        }
        transformer = NullTransformer()

        expected_result = pd.DataFrame({
            'age': [62.0, 53.0, 53.0, 45.0, 53.25],
            '?age': [1, 1, 1, 1, 0]
        })

        # Run
        result = transformer.fit_transform(col, col_meta)

        # Check
        assert result.equals(expected_result)

    def test_reverse_transform(self):
        """Checks the conversion of the data back into original format."""

        # Setup
        col_meta = {
            'name': 'age',
            'type': 'number'
        }
        transformer = NullTransformer()
        data = pd.DataFrame({
            'age': [62, 35, 0, 24, 27],
            '?age': [1, 1, 0, 1, 1]
        })

        expected_result = pd.Series([62, 35, np.nan, 24, 27], name='age')

        # Result
        result = transformer.reverse_transform(data, col_meta)

        # Check
        assert result.age.equals(expected_result)
