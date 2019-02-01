from unittest import TestCase

import numpy as np
import pandas as pd

from rdt.transformers.datetime import DTTransformer
from tests import safe_compare_dataframes


class TestDTTransformer(TestCase):
    def setUp(self):
        self.normal_data = pd.read_csv('tests/data/datetime/normal.csv')
        self.missing_data = pd.read_csv('tests/data/datetime/missing.csv')

        self.normal_meta = {
            "name": "date_account_created",
            "type": "datetime",
            "format": "%m/%d/%y"
        }
        self.missing_meta = {
            "name": "date_first_booking",
            "type": "datetime",
            "format": "%m/%d/%y"
        }

    def test_fit_transform(self):
        """fit_transforms transforms datetime values into floats."""
        # Setup
        transformer = DTTransformer(self.normal_meta)
        expected_result = pd.Series([
            1.3885524e+18,
            1.3886388e+18,
            1.3887252e+18,
            1.3888116e+18
        ])
        column_name = self.normal_meta['name']

        # Run
        transformed = transformer.fit_transform(self.normal_data)
        result = transformed[column_name]

        # Check
        assert np.allclose(result, expected_result, 1e-03)

    def test_fit_transform_out_of_bounds(self):
        """Out of bounds values should be transformed too."""
        # Setup
        out_of_bounds_data = pd.DataFrame({
            'date_first_booking': [
                '2262-04-11 23:47:16.854775',
                '2263-04-11 23:47:16.854776'
            ]
        })

        expected_result = pd.Series(
            [9223372036854774784, np.nan],
            name='date_first_booking'
        )

        out_of_bounds_meta = {
            'name': 'date_first_booking',
            'type': 'datetime',
            'format': '%Y-%m-%d %H:%M:%S.%f'
        }

        # Run
        transformer = DTTransformer(out_of_bounds_meta)
        transformed = transformer.fit_transform(out_of_bounds_data)
        result = transformed[self.missing_meta['name']]

        # Check
        assert safe_compare_dataframes(result, expected_result)

    def test_reverse_transform(self):
        """reverse_transform reverse fit_transforms."""
        # Setup
        transformer = DTTransformer(self.normal_meta)
        transformed = transformer.fit_transform(self.normal_data)

        # Run
        result = transformer.reverse_transform(transformed)

        # Check
        assert result.equals(self.normal_data)

    def test_reverse_transform_missing(self):
        """Missing values are left unchanged."""
        # Setup
        transformer = DTTransformer(self.missing_meta)
        transformed = transformer.fit_transform(self.missing_data)
        expected_result = self.missing_data['date_first_booking']

        # Run
        result = transformer.reverse_transform(transformed)[transformer.col_name]

        # Check
        assert safe_compare_dataframes(result, expected_result)

    def test_reversibility_transforms(self):
        """Transforming and reverse transforming a column leaves it unchanged."""
        # Setup
        transformer = DTTransformer(self.normal_meta)

        # Run
        transformed_data = transformer.fit_transform(self.normal_data)
        reversed_data = transformer.reverse_transform(transformed_data)

        # Check
        assert reversed_data.equals(self.normal_data)
