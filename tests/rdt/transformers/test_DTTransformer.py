from unittest import TestCase, skipIf

import numpy as np
import pandas as pd

from rdt.transformers.DTTransformer import DTTransformer
from rdt.utils import get_col_info

# SKIPPED TESTS
TESTS_WITH_DATA = True


class DTTransformerTest(TestCase):
    def setUp(self):
        self.normal_data = pd.read_csv('tests/data/normal_datetime.csv')
        self.missing_data = pd.read_csv('tests/data/missing_datetime.csv')
        self.normal_meta = {
            "name": "date_account_created",
            "type": "datetime",
            "format": "%m/%d/%y",
        }
        self.missing_meta = {
            "name": "date_first_booking",
            "type": "datetime",
            "format": "%m/%d/%y",
        }
        self.normal_data = self.normal_data[self.normal_meta['name']]
        self.missing_data = self.missing_data[self.missing_meta['name']]
        self.transformer = DTTransformer()

    def test_fit_transform(self):
        # get truncated column
        result = pd.Series(
            [
                1.3885524e+18,
                1.3886388e+18,
                1.3887252e+18,
                1.3888116e+18
            ],
            name=self.normal_meta['name']
        )

        transformed = self.transformer.fit_transform(self.normal_data,
                                                     self.normal_meta)
        predicted = transformed[self.normal_meta['name']]
        # load correct answer
        self.assertTrue(np.allclose(result, predicted, 1e-03))

    def test_reverse_transform(self):
        """ """

        # Setup
        raw = pd.Series([
                '01/01/14',
                '01/02/14',
                '01/03/14',
                '01/04/14',
            ], name='date_account_created'
        )
        transformed = self.transformer.fit_transform(raw, self.normal_meta)
        transformed = transformed[self.normal_meta['name']]

        # Run
        result = self.transformer.reverse_transform(
            transformed, self.normal_meta, missing=False)

        # Check
        result = result[self.normal_meta['name']]
        expected_result = raw

        assert result.equals(expected_result)

    def test_fit_transform_missing(self):
        # get truncated column
        result = pd.Series([np.nan,
                            1.4187924e+18,
                            np.nan,
                            1.3886388e+18,
                            1.3996944e+18],
                           name=self.missing_meta['name'])
        transformed = self.transformer.fit_transform(self.missing_data,
                                                     self.missing_meta)
        predicted = transformed[self.missing_meta['name']]
        for i in range(len(result)):
            if not np.isnan(result[i]):
                self.assertTrue(np.allclose(result[i],
                                            predicted[i],
                                            1e-03))
            else:
                self.assertFalse(np.isnan(predicted[i]))

    def test_reverse_transform_missing(self):
        transformed = self.transformer.fit_transform(self.missing_data,
                                                     self.missing_meta)
        predicted = self.transformer.reverse_transform(transformed,
                                                       self.missing_meta)
        predicted = predicted[self.missing_meta['name']]
        result = self.missing_data
        for i in range(len(result)):
            if isinstance(result[i], str):
                res_date = result[i].split('/')
                pred_date = predicted[i].split('/')
                for j in range(len(res_date)):
                    self.assertEqual(int(res_date[j]),
                                     int(pred_date[j]))
            else:
                self.assertFalse(predicted[i] == predicted[i])

    @skipIf(TESTS_WITH_DATA, 'demo_downloader should have been run.')
    def test_reversibility_transforms(self):
        """Transforming and reverse transforming a column leaves it unchanged."""
        # Setup
        col, col_meta = get_col_info(
            'users', 'date_account_created', 'demo/Airbnb_demo_meta.json')

        transformer = DTTransformer()

        # Run
        transformed_data = transformer.fit_transform(col, col_meta)
        reversed_data = transformer.reverse_transform(transformed_data, col_meta)

        # Check
        assert col.dtype == object
        assert transformed_data['date_account_created'].dtype == float
        assert reversed_data['date_account_created'].equals(col)
