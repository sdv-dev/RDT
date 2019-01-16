from unittest import TestCase

import numpy as np
import pandas as pd

from rdt.transformers.datetime import DTTransformer


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
        transformer = DTTransformer(out_of_bounds_meta)
        transformed = transformer.fit_transform(out_of_bounds_data)
        result = transformed[self.missing_meta['name']]

        # Check
        self.assertTrue(np.allclose(result, expected_result, 1e-03))

    def test_reverse_transform(self):
        """reverse_transform reverse fit_transforms."""
        # Setup
        transformer = DTTransformer(self.normal_meta, missing=False)
        transformed = transformer.fit_transform(self.normal_data)

        # Run
        result = transformer.reverse_transform(transformed)

        # Check
        assert result.equals(self.normal_data)

    def test_reverse_transform_nan_dataframe(self):
        """Checks that nans are handled correctly in reverse transformation"""

        # Setup
        transformer = DTTransformer(self.normal_meta, missing=False)
        transformer.fit_transform(self.normal_data)
        transformed = pd.DataFrame({
            'date_account_created': [
                1.3885524e+18,
                1.3885524e+18,
                1.3887252e+18,
                1.3887252e+18,
                np.nan
            ],
        })
        expected_result = pd.DataFrame({'date_account_created': [
            '01/01/14',
            '01/01/14',
            '01/03/14',
            '01/03/14',
            '01/01/14'
        ]})

        # Run
        result = transformer.reverse_transform(transformed)

        # Check
        assert result.equals(expected_result)

    def test_reverse_transform_nan_series(self):
        """Checks that nans are handled correctly in reverse transformation"""

        # Setup
        transformer = DTTransformer(self.normal_meta, missing=False)
        transformed = pd.Series(
            [
                1.3885524e+18,
                1.3885524e+18,
                1.3887252e+18,
                1.3887252e+18,
                np.nan
            ], name='date_account_created'
        )
        expected_result = pd.DataFrame({
            'date_account_created': [
                '01/01/14',
                '01/01/14',
                '01/03/14',
                '01/03/14',
                '01/01/14'
            ]
        })
        transformer.fit_transform(self.normal_data)

        # Run
        result = transformer.reverse_transform(transformed)

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
        transformer = DTTransformer(self.missing_meta)

        # Run
        transformed = transformer.fit_transform(self.missing_data)
        result = transformed[column_name]

        # Check
        # There are no null values in the transformed data.
        assert not result.isnull().any().any()
        assert np.isclose(expected_result, result).all()

    def test_reverse_transform_missing(self):
        # Setup
        transformer = DTTransformer(self.missing_meta)
        transformed = transformer.fit_transform(self.missing_data)
        expected_result = self.missing_data['date_first_booking']

        # Run
        result = transformer.reverse_transform(transformed)[transformer.col_name]

        # Check
        assert result.equals(expected_result)

    def test_reversibility_transforms(self):
        """Transforming and reverse transforming a column leaves it unchanged."""
        # Setup
        transformer = DTTransformer(self.normal_meta, missing=False)

        # Run
        transformed_data = transformer.fit_transform(self.normal_data)
        reversed_data = transformer.reverse_transform(transformed_data)

        # Check
        assert reversed_data.equals(self.normal_data)
