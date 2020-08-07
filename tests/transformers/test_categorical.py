from unittest import TestCase
from unittest.mock import Mock, patch

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

    def test__get_faker_anonymize_tuple(self):
        """Test _get_faker when anonymize is a tuple"""
        # Setup

        # Run
        transformer = Mock()
        transformer.anonymize = ('email',)

        result = CategoricalTransformer._get_faker(transformer)

        # Asserts
        self.assertEqual(
            result.__name__,
            'faker',
            "Expected faker function"
        )

    def test__get_faker_anonymize_list(self):
        """Test _get_faker when anonymize is a list"""
        # Run
        transformer = Mock()
        transformer.anonymize = ['email']

        result = CategoricalTransformer._get_faker(transformer)

        # Asserts
        self.assertEqual(
            result.__name__,
            'faker',
            "Expected faker function"
        )

    def test__get_faker_anonymize_list_type(self):
        """Test _get_faker when anonymize is a list with two elements"""
        # Run
        transformer = Mock()
        transformer.anonymize = ['credit_card_number', 'visa']

        faker_method = CategoricalTransformer._get_faker(transformer)
        fake_value = faker_method()

        # Asserts
        assert isinstance(fake_value, str)
        assert len(fake_value) == 16

    def test__get_faker_anonymize_not_tuple_or_list(self):
        """Test _get_faker when anonymize is neither a typle or a list"""
        # Run
        transformer = Mock()
        transformer.anonymize = 'email'

        result = CategoricalTransformer._get_faker(transformer)

        # Asserts
        self.assertEqual(
            result.__name__,
            'faker',
            "Expected faker function"
        )

    def test__get_faker_anonymize_category_not_exist(self):
        """Test _get_faker with a category that don't exist"""
        # Run & assert
        transformer = Mock()
        transformer.anonymize = 'SuP3R-P1Th0N-P0w3R'

        with self.assertRaises(ValueError):
            CategoricalTransformer._get_faker(transformer)

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

        assert transformer._get_faker.call_count == 1
        self.assertEqual(
            len(result),
            expect_result_len,
            "Length of anonymized data unexpected"
        )

    def test__get_intervals(self):
        """Test get category intervals"""
        # Setup
        data = pd.Series(['bar', 'foo', 'foo', 'tar'])

        # Run
        result = CategoricalTransformer._get_intervals(data)

        # Asserts
        expected_intervals = {
            'foo': (0, 0.5, 0.25, 0.5 / 6),
            'tar': (0.5, 0.75, 0.625, 0.25 / 6),
            'bar': (0.75, 1, 0.875, 0.25 / 6)
        }
        assert result == expected_intervals

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
        expect_intervals_call_count = 1
        expect_intervals_call_args = pd.Series(['bar', 'foo', 'foo', 'tar'])

        self.assertEqual(
            transformer._anonymize.call_count,
            expect_anonymize_call_count,
            "Anonymize must be called only when anonymize is something"
        )

        self.assertEqual(
            transformer._get_intervals.call_count,
            expect_intervals_call_count,
            "Get intervals will be called always in fit"
        )

        pd.testing.assert_series_equal(
            transformer._get_intervals.call_args[0][0],
            expect_intervals_call_args
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
        expect_intervals_call_count = 1
        expect_intervals_call_args = pd.Series(['bar', 'foo', 'foo', 'tar'])

        self.assertEqual(
            transformer._anonymize.call_count,
            expect_anonymize_call_count,
            "Anonymize must be called only when anonymize is something"
        )

        self.assertEqual(
            transformer._get_intervals.call_count,
            expect_intervals_call_count,
            "Get intervals will be called always in fit"
        )

        pd.testing.assert_series_equal(
            transformer._get_intervals.call_args[0][0],
            expect_intervals_call_args
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
        expect_intervals_call_count = 1
        expect_intervals_call_args = pd.Series(['bar', 'foo', 'foo', 'tar'])

        self.assertEqual(
            transformer._anonymize.call_count,
            expect_anonymize_call_count,
            "Anonymize must be called only once"
        )

        self.assertEqual(
            transformer._get_intervals.call_count,
            expect_intervals_call_count,
            "Get intervals will be called always in fit"
        )

        pd.testing.assert_series_equal(
            transformer._get_intervals.call_args[0][0],
            expect_intervals_call_args
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
        expect_intervals_call_count = 1
        expect_intervals_call_args = pd.Series(['bar', 'foo', 'foo', 'tar'])

        self.assertEqual(
            transformer._anonymize.call_count,
            expect_anonymize_call_count,
            "Anonymize must be called only once"
        )

        self.assertEqual(
            transformer._get_intervals.call_count,
            expect_intervals_call_count,
            "Get intervals will be called always in fit"
        )

        pd.testing.assert_series_equal(
            transformer._get_intervals.call_args[0][0],
            expect_intervals_call_args
        )

    def test__get_value_no_fuzzy(self):
        # Run
        transformer = Mock()
        transformer.fuzzy = False
        transformer.intervals = {
            'foo': (0, 0.5, 0.25, 0.5 / 6),
        }

        result = CategoricalTransformer._get_value(transformer, 'foo')

        # Asserts
        assert result == 0.25

    @patch('scipy.stats.norm.rvs')
    def test__get_value_fuzzy(self, rvs_mock):
        # setup
        rvs_mock.return_value = 0.2745

        # Run
        transformer = Mock()
        transformer.fuzzy = True
        transformer.intervals = {
            'foo': (0, 0.5, 0.25, 0.5 / 6),
        }

        result = CategoricalTransformer._get_value(transformer, 'foo')

        # Asserts
        assert result == 0.2745

    @patch('rdt.transformers.categorical.MAPS', new_callable=dict)
    def test_transform_array_anonymize(self, mock_maps):
        """Test transform a numpy.array, anonymize"""
        # Setup
        data = np.array(['bar', 'foo', 'foo', 'tar'])

        # Run
        transformer = Mock()
        transformer.anonymize = 'email'
        transformer.intervals = [1, 2, 3]

        mock_maps[id(transformer)] = {
            'bar': 'bar_x',
            'foo': 'foo_x',
            'tar': 'tar_x'
        }

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
        transformer.intervals = [1, 2, 3]

        CategoricalTransformer.transform(transformer, data)

        # Asserts
        expect_maps_call_count = 0

        self.assertEqual(
            mock_maps.call_count,
            expect_maps_call_count,
            "Dont call to the map encoder when not anonymize"
        )

    def test__normalize_no_clip(self):
        """Test normalize data"""
        # Setup
        data = pd.Series([-0.43, 0.1234, 1.5, -1.31])

        transformer = Mock()
        transformer.clip = False

        # Run
        result = CategoricalTransformer._normalize(transformer, data)

        # Asserts
        expect = pd.Series([0.57, 0.1234, 0.5, 0.69], dtype=float)

        pd.testing.assert_series_equal(result, expect)

    def test__normalize_clip(self):
        """Test normalize data with clip=True"""
        # Setup
        data = pd.Series([-0.43, 0.1234, 1.5, -1.31])

        transformer = Mock()
        transformer.clip = True

        # Run
        result = CategoricalTransformer._normalize(transformer, data)

        # Asserts
        expect = pd.Series([0.0, 0.1234, 1.0, 0.0], dtype=float)

        pd.testing.assert_series_equal(result, expect)

    def test_reverse_transform_array(self):
        """Test reverse_transform a numpy.array"""
        # Setup
        data = np.array([-0.6, 0.2, 0.6, -0.2])
        normalized_data = pd.Series([0.4, 0.2, 0.6, 0.8])

        intervals = {
            'foo': (0, 0.5),
            'bar': (0.5, 0.75),
            'tar': (0.75, 1),
        }

        # Run
        transformer = Mock()
        transformer._normalize.return_value = normalized_data
        transformer.intervals = intervals

        result = CategoricalTransformer.reverse_transform(transformer, data)

        # Asserts
        expect = pd.Series(['foo', 'foo', 'bar', 'tar'])

        pd.testing.assert_series_equal(result, expect)
