import json
from unittest import TestCase

import numpy as np
import pandas as pd

from rdt.transformers.DTTransformer import DTTransformer


class TestDTTransformer(TestCase):
    def setUp(self):
        self.normal_data = pd.read_csv('tests/data/datetime/normal.csv')
        self.missing_data = pd.read_csv('tests/data/datetime/missing.csv')

        with open('tests/data/datetime/normal.json') as f:
            self.normal_meta = json.load(f)

        with open('tests/data/datetime/missing.json') as f:
            self.missing_meta = json.load(f)

        self.transformer = DTTransformer()

    def test_fit_transform(self):
        """fit_transforms transforms datetime values into floats."""
        # Setup
        transformer = DTTransformer()
        expected_result = pd.Series([
            1.3885524e+18,
            1.3886388e+18,
            1.3887252e+18,
            1.3888116e+18
        ])
        column_name = self.normal_meta['name']

        # Run
        transformed = transformer.fit_transform(self.normal_data, self.normal_meta)
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

        expected_result = pd.Series([
            9.223386e+18,
            0.000000e+00
        ])

        out_of_bounds_meta = {
            "name": "date_first_booking",
            "type": "datetime",
            "format": "%Y-%m-%d %H:%M:%S.%f"
        }

        # Run
        transformed = self.transformer.fit_transform(out_of_bounds_data, out_of_bounds_meta)
        result = transformed[self.missing_meta['name']]

        # Check
        self.assertTrue(np.allclose(result, expected_result, 1e-03))

    def test_reverse_transform(self):
        """reverse_transform reverse fit_transforms."""
        # Setup
        column_name = self.normal_meta['name']
        transformed = self.transformer.fit_transform(
            self.normal_data, self.normal_meta, missing=False)

        # Run
        result = self.transformer.reverse_transform(
            transformed[column_name], self.normal_meta, missing=False)

        # Check
        assert result.equals(self.normal_data)

    def test_reverse_transform_nan(self):
        """Checks that nans are handled correctly in reverse transformation"""

        # Setup
        transformed = pd.Series(
            [
                1.3885524e+18,
                1.3885524e+18,
                1.3887252e+18,
                1.3887252e+18,
                np.nan
            ],
            name='date_account_created'
        )
        expected_result = pd.DataFrame({'date_account_created': [
            '01/01/14',
            '01/01/14',
            '01/03/14',
            '01/03/14',
            '01/01/14'
        ]})
        self.transformer.fit_transform(self.normal_data, self.normal_meta)

        # Run
        result = self.transformer.reverse_transform(transformed, self.normal_meta, False)

        # Check
        assert result.equals(expected_result)

    def test_fit_transform_missing(self):
        """fit_transform will fill NaN values by default."""
        # Setup
        column_name = self.missing_meta['name']
        expected_result = pd.Series(
            [
                0.00000000e+00,
                1.41877080e+18,
                0.00000000e+00,
                1.38861720e+18,
                1.39967280e+18
            ],
            name=column_name
        )

        # Run
        transformed = self.transformer.fit_transform(self.missing_data, self.missing_meta)
        result = transformed[column_name]

        # Check
        # There are no null values in the transformed data.
        assert not result.isnull().any().any()
        assert np.isclose(expected_result, result).all()

    def test_reverse_transform_missing(self):
        # Setup
        transformed = self.transformer.fit_transform(self.missing_data, self.missing_meta)
        expected_result = self.missing_data['date_first_booking']

        # Run
        result = self.transformer.reverse_transform(transformed, self.missing_meta)
        result = result[self.missing_meta['name']]

        # Check
        assert result.equals(expected_result)

    def test_reversibility_transforms(self):
        """Transforming and reverse transforming a column leaves it unchanged."""
        # Setup
        transformer = DTTransformer()

        # Run
        transformed_data = transformer.fit_transform(self.normal_data, self.normal_meta)
        reversed_data = transformer.reverse_transform(transformed_data, self.normal_meta)

        # Check
        assert reversed_data.equals(self.normal_data)
