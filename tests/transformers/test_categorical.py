from unittest import TestCase
from unittest.mock import Mock, call, patch

import numpy as np
import pandas as pd

from rdt.transformers import CategoricalTransformer


class TestCategoricalTransformer(TestCase):

    def test___init__(self):
        """Test default instance"""
        # Run
        transformer = CategoricalTransformer()

        # Asserts
        self.assertFalse(transformer.anonymize, "Unexpected anonimyze default value")

    def test_get_faker_anonymize_tuple(self):
        """Test get_faker when anonymize is a tuple"""
        # Setup

        # Run
        transformer = Mock()
        transformer.anonymize = ('email',)

        result = CategoricalTransformer.get_faker(transformer)

        # Asserts
        self.assertEqual(
            result.__name__,
            'faker',
            "Expected faker function"
        )

    def test_get_faker_anonymize_list(self):
        """Test get_faker when anonymize is a list"""
        # Run
        transformer = Mock()
        transformer.anonymize = ['email']

        result = CategoricalTransformer.get_faker(transformer)

        # Asserts
        self.assertEqual(
            result.__name__,
            'faker',
            "Expected faker function"
        )

    def test_get_faker_anonymize_not_tuple_or_list(self):
        """Test get_faker when anonymize is neither a typle or a list"""
        # Run
        transformer = Mock()
        transformer.anonymize = 'email'

        result = CategoricalTransformer.get_faker(transformer)

        # Asserts
        self.assertEqual(
            result.__name__,
            'faker',
            "Expected faker function"
        )

    def test_get_faker_anonymize_category_not_exist(self):
        """Test get_faker with a category that don't exist"""
        # Run & assert
        transformer = Mock()
        transformer.anonymize = 'SuP3R-P1Th0N-P0w3R'

        with self.assertRaises(ValueError):
            CategoricalTransformer.get_faker(transformer)

    def test__anonymize(self):
        """Test anonymize data"""
        # Setup
        category = 'email'

        data = pd.Series(['foo', 'bar', 'foo', 'tar'])

        # Run
        transformer = Mock()
        transformer.anonymize = category

        result = CategoricalTransformer._anonymize(transformer, data)

        # Asserts
        expect_result_len = 4

        self.assertEqual(
            len(result),
            expect_result_len,
            "Length of anonymized data unexpected"
        )

    def test__get_probabilities(self):
        """Test get category probabilities"""
        # Setup
        data = pd.Series(['bar', 'foo', 'foo', 'tar'])

        # Run
        result = CategoricalTransformer._get_probabilities(data)

        # Asserts
        # Intervals (item[0]) are compared with the difference because the order may change
        expect = {
            'foo': (0.5, 0.25, 0.08333),
            'bar': (0.25, 0.625, 0.04167),
            'tar': (0.25, 0.875, 0.04167)
        }

        for key in result.keys():
            self.assertEqual(
                result[key][0][1] - result[key][0][0],
                expect[key][0],
                "Wrong interval: {}".format(key))

            # REVIEW:
            # can't assert when different means are computed in different orders
            # self.assertEqual(
            #     result[key][1],
            #     expect[key][1],
            #     "Wrong mean: {}".format(key)
            # )

            self.assertEqual(
                round(result[key][2], 5),
                expect[key][2],
                "Wrong std: {}".format(key)
            )

    def test_fit_array_no_anonymize(self):
        """Test fit with a numpy.array, don't anonymize"""
        # Setup
        data = np.array(['bar', 'foo', 'foo', 'tar'])

        # Run
        transformer = Mock()
        transformer.anonymize = None

        CategoricalTransformer.fit(transformer, data)

        # Asserts
        expect_anonymize_call_count = 0
        expect_probabilities_call_count = 1
        expect_probabilities_call_args = pd.Series(['bar', 'foo', 'foo', 'tar'])

        self.assertEqual(
            transformer._anonymize.call_count,
            expect_anonymize_call_count,
            "Anonymize must be called only when anonymize is something"
        )

        self.assertEqual(
            transformer._get_probabilities.call_count,
            expect_probabilities_call_count,
            "Get probabilities will be called always in fit"
        )

        pd.testing.assert_series_equal(
            transformer._get_probabilities.call_args[0][0],
            expect_probabilities_call_args
        )

    def test_fit_series_no_anonymize(self):
        """Test fit with a pandas.Series, don't anonymize"""
        # Setup
        data = pd.Series(['bar', 'foo', 'foo', 'tar'])

        # Run
        transformer = Mock()
        transformer.anonymize = None

        CategoricalTransformer.fit(transformer, data)

        # Asserts
        expect_anonymize_call_count = 0
        expect_probabilities_call_count = 1
        expect_probabilities_call_args = pd.Series(['bar', 'foo', 'foo', 'tar'])

        self.assertEqual(
            transformer._anonymize.call_count,
            expect_anonymize_call_count,
            "Anonymize must be called only when anonymize is something"
        )

        self.assertEqual(
            transformer._get_probabilities.call_count,
            expect_probabilities_call_count,
            "Get probabilities will be called always in fit"
        )

        pd.testing.assert_series_equal(
            transformer._get_probabilities.call_args[0][0],
            expect_probabilities_call_args
        )

    def test_fit_array_anonymize(self):
        """Test fit with a numpy.array, anonymize"""
        # Setup
        data = np.array(['bar', 'foo', 'foo', 'tar'])
        data_anonymized = pd.Series(['bar', 'foo', 'foo', 'tar'])

        # Run
        transformer = Mock()
        transformer.anonymize = 'email'
        transformer._anonymize.return_value = data_anonymized

        CategoricalTransformer.fit(transformer, data)

        # Asserts
        expect_anonymize_call_count = 1
        expect_probabilities_call_count = 1
        expect_probabilities_call_args = pd.Series(['bar', 'foo', 'foo', 'tar'])

        self.assertEqual(
            transformer._anonymize.call_count,
            expect_anonymize_call_count,
            "Anonymize must be called only once"
        )

        self.assertEqual(
            transformer._get_probabilities.call_count,
            expect_probabilities_call_count,
            "Get probabilities will be called always in fit"
        )

        pd.testing.assert_series_equal(
            transformer._get_probabilities.call_args[0][0],
            expect_probabilities_call_args
        )

    def test_fit_series_anonymize(self):
        """Test fit with a pandas.Series, anonymize"""
        # Setup
        data = pd.Series(['bar', 'foo', 'foo', 'tar'])
        data_anonymized = pd.Series(['bar', 'foo', 'foo', 'tar'])

        # Run
        transformer = Mock()
        transformer.anonymize = 'email'
        transformer._anonymize.return_value = data_anonymized

        CategoricalTransformer.fit(transformer, data)

        # Asserts
        expect_anonymize_call_count = 1
        expect_probabilities_call_count = 1
        expect_probabilities_call_args = pd.Series(['bar', 'foo', 'foo', 'tar'])

        self.assertEqual(
            transformer._anonymize.call_count,
            expect_anonymize_call_count,
            "Anonymize must be called only once"
        )

        self.assertEqual(
            transformer._get_probabilities.call_count,
            expect_probabilities_call_count,
            "Get probabilities will be called always in fit"
        )

        pd.testing.assert_series_equal(
            transformer._get_probabilities.call_args[0][0],
            expect_probabilities_call_args
        )

    @patch('scipy.stats.norm.rvs')
    def test_get_val(self, scipy_mock):
        """Test convert category value into num between 0 and 1"""
        # Run
        transformer = Mock()
        transformer.probabilities = {
            'foo': ((0, 0.5), 0.25, 0.08333)
        }

        CategoricalTransformer.get_val(transformer, 'foo')

        # Asserts
        expect_call_count = 1
        expect_call_args = call(0.25, 0.08333)

        self.assertEqual(
            scipy_mock.call_count,
            expect_call_count
        )

        self.assertEqual(
            scipy_mock.call_args,
            expect_call_args
        )

    @patch('rdt.transformers.categorical.MAPS')
    def test_transform_array_anonymize(self, mock_maps):
        """Test transform a numpy.array, anonymize"""
        # Setup
        mock_maps = {}

        data = np.array(['bar', 'foo', 'foo', 'tar'])

        # Run
        transformer = Mock()
        transformer.anonymize = 'email'

        mock_maps[id(transformer)] = data = np.array(['bar_x', 'foo_x', 'foo_x', 'tar_x'])

        result = CategoricalTransformer.transform(transformer, data)

        # Asserts
        expect_result_len = 4

        self.assertEqual(
            len(result),
            expect_result_len,
            "Unexpected length of transformed data"
        )

    @patch('rdt.transformers.categorical.MAPS')
    def test_transform_array_no_anonymize(self, mock_maps):
        """Test transform a numpy.array, no anonymize"""
        # Setup
        data = np.array(['bar', 'foo', 'foo', 'tar'])

        # Run
        transformer = Mock()
        transformer.anonymize = None

        CategoricalTransformer.transform(transformer, data)

        # Asserts
        expect_maps_call_count = 0

        self.assertEqual(
            mock_maps.call_count,
            expect_maps_call_count,
            "Dont call to the map encoder when not anonymize"
        )

    def test__normalize(self):
        """Test normalize data"""
        # Setup
        data = pd.Series([-0.43, 0.1234, 1.5, -1.31])

        # Run
        result = CategoricalTransformer._normalize(data)

        # Asserts
        expect = pd.Series([0.57, 0.1234, 0.5, 0.69], dtype=float)

        pd.testing.assert_series_equal(result, expect)

    def test_reverse_transform_array(self):
        """Test reverse_transform a numpy.array"""
        # Setup
        data = np.array([-0.6, 0.2, 0.6, -0.2])
        normalized_data = pd.Series([0.4, 0.2, 0.6, 0.8])

        probabilities = {
            'foo': ((0, 0.5), None, None),
            'bar': ((0.5, 0.75), None, None),
            'tar': ((0.75, 1), None, None)
        }

        # Run
        transformer = Mock()
        transformer._normalize.return_value = normalized_data
        transformer.probabilities = probabilities

        result = CategoricalTransformer.reverse_transform(transformer, data)

        # Asserts
        expect = pd.Series(['foo', 'foo', 'bar', 'tar'])

        pd.testing.assert_series_equal(result, expect)
