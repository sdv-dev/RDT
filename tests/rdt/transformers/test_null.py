import unittest

import numpy as np
import pandas as pd

from rdt.transformers.null import NullTransformer


class TestNullTransformer(unittest.TestCase):
    def test___init__(self):
        """On __init__ set type to number and datetime."""
        # Setup
        column_metadata = {
            'name': 'age',
            'type': 'number'
        }

        # Run
        transformer = NullTransformer(column_metadata)

        # Check
        assert transformer.type == ['datetime', 'number', 'categorical']

    def test_fit_transform_isnull(self):
        """It will replace nan values with 0 and creats a new column."""

        # Setup
        col = pd.Series([62, np.nan, np.nan, np.nan, np.nan], name='age')
        column_metadata = {
            'name': 'age',
            'type': 'number'
        }
        transformer = NullTransformer(column_metadata)

        expected_result = pd.DataFrame(
            {
                'age': [62.0, 62.0, 62.0, 62.0, 62.0],
                '?age': [1, 0, 0, 0, 0]
            },
            columns=['age', '?age']
        )

        # Run
        result = transformer.fit_transform(col)

        # Check
        assert result.equals(expected_result)

    def test_fit_transform_notnull(self):
        """Creates a new column with the mean of the values."""

        # Setup
        col = pd.Series([62, 53, 53, 45, np.nan])
        column_metadata = {
            'name': 'age',
            'type': 'number'
        }
        transformer = NullTransformer(column_metadata)

        expected_result = pd.DataFrame(
            {
                'age': [62.0, 53.0, 53.0, 45.0, 53.25],
                '?age': [1, 1, 1, 1, 0]
            },
            columns=['age', '?age']
        )

        # Run
        result = transformer.fit_transform(col)

        # Check
        assert result.equals(expected_result)

    def test_reverse_transform(self):
        """Checks the conversion of the data back into original format."""

        # Setup
        column_metadata = {
            'name': 'age',
            'type': 'number'
        }
        transformer = NullTransformer(column_metadata)
        data = pd.DataFrame({
            'age': [62, 35, 0, 24, 27],
            '?age': [1, 1, 0, 1, 1]
        })

        expected_result = pd.Series([62, 35, np.nan, 24, 27], name='age')

        # Result
        result = transformer.reverse_transform(data)

        # Check
        assert result.age.equals(expected_result)
