import re
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from rdt.transformers import CategoricalTransformer, OneHotEncodingTransformer

RE_SSN = re.compile(r'\d\d\d-\d\d-\d\d\d\d')


class TestCategoricalTransformer:

    def test___init__(self):
        """Passed arguments must be stored as attributes."""
        # Run
        transformer = CategoricalTransformer(
            fuzzy='fuzzy_value',
            clip='clip_value',
        )

        # Asserts
        assert transformer.fuzzy == 'fuzzy_value'
        assert transformer.clip == 'clip_value'

    def test__get_intervals(self):
        # Run
        data = pd.Series(['bar', 'foo', 'foo', 'tar'])
        result = CategoricalTransformer._get_intervals(data)

        # Asserts
        expected_intervals = {
            'foo': (0, 0.5, 0.25, 0.5 / 6),
            'tar': (0.5, 0.75, 0.625, 0.25 / 6),
            'bar': (0.75, 1, 0.875, 0.25 / 6)
        }
        assert result == expected_intervals

    def test_fit(self):
        # Setup
        transformer = CategoricalTransformer()

        # Run
        data = np.array(['bar', 'foo', 'foo', 'tar'])
        transformer.fit(data)

        # Asserts
        expected_intervals = {
            'foo': (0, 0.5, 0.25, 0.5 / 6),
            'tar': (0.5, 0.75, 0.625, 0.25 / 6),
            'bar': (0.75, 1, 0.875, 0.25 / 6)
        }
        assert transformer.intervals == expected_intervals

    def test__get_value_no_fuzzy(self):
        # Setup
        transformer = CategoricalTransformer(fuzzy=False)
        transformer.fuzzy = False
        transformer.intervals = {
            'foo': (0, 0.5, 0.25, 0.5 / 6),
        }

        # Run
        result = transformer._get_value('foo')

        # Asserts
        assert result == 0.25

    @patch('scipy.stats.norm.rvs')
    def test__get_value_fuzzy(self, rvs_mock):
        # setup
        rvs_mock.return_value = 0.2745

        transformer = CategoricalTransformer(fuzzy=True)
        transformer.intervals = {
            'foo': (0, 0.5, 0.25, 0.5 / 6),
        }

        # Run
        result = transformer._get_value('foo')

        # Asserts
        assert result == 0.2745

    def test__normalize_no_clip(self):
        """Test normalize data"""
        # Setup
        transformer = CategoricalTransformer(clip=False)

        # Run
        data = pd.Series([-0.43, 0.1234, 1.5, -1.31])
        result = transformer._normalize(data)

        # Asserts
        expect = pd.Series([0.57, 0.1234, 0.5, 0.69], dtype=float)

        pd.testing.assert_series_equal(result, expect)

    def test__normalize_clip(self):
        """Test normalize data with clip=True"""
        # Setup
        transformer = CategoricalTransformer(clip=True)

        # Run
        data = pd.Series([-0.43, 0.1234, 1.5, -1.31])
        result = transformer._normalize(data)

        # Asserts
        expect = pd.Series([0.0, 0.1234, 1.0, 0.0], dtype=float)

        pd.testing.assert_series_equal(result, expect)

    def test_reverse_transform_array(self):
        """Test reverse_transform a numpy.array"""
        # Setup
        transformer = CategoricalTransformer()
        transformer.dtype = object
        transformer.intervals = {
            'foo': (0, 0.5),
            'bar': (0.5, 0.75),
            'tar': (0.75, 1),
        }

        # Run
        data = np.array([-0.6, 0.2, 0.6, -0.2])
        result = transformer.reverse_transform(data)

        # Asserts
        expect = pd.Series(['foo', 'foo', 'bar', 'tar'])

        pd.testing.assert_series_equal(result, expect)

    def test_reversible_strings(self):
        data = pd.Series(['a', 'b', 'a', 'c'])
        transformer = CategoricalTransformer()

        reverse = transformer.reverse_transform(transformer.fit_transform(data))

        pd.testing.assert_series_equal(data, reverse)

    def test_reversible_strings_2_categories(self):
        data = pd.Series(['a', 'b', 'a', 'b'])
        transformer = CategoricalTransformer()

        reverse = transformer.reverse_transform(transformer.fit_transform(data))

        pd.testing.assert_series_equal(data, reverse)

    def test_reversible_integers(self):
        data = pd.Series([1, 2, 3, 2])
        transformer = CategoricalTransformer()

        reverse = transformer.reverse_transform(transformer.fit_transform(data))

        pd.testing.assert_series_equal(data, reverse)

    def test_reversible_bool(self):
        data = pd.Series([True, False, True, False])
        transformer = CategoricalTransformer()

        reverse = transformer.reverse_transform(transformer.fit_transform(data))

        pd.testing.assert_series_equal(data, reverse)

    def test_reversible_mixed(self):
        data = pd.Series([True, 'a', 1, None])
        transformer = CategoricalTransformer()

        reverse = transformer.reverse_transform(transformer.fit_transform(data))

        pd.testing.assert_series_equal(data, reverse)


class TestOneHotEncodingTransformer:

    def test__prepare_data_empty_lists(self):
        # Setup
        ohet = OneHotEncodingTransformer()
        data = [[], [], []]

        # Assert
        with pytest.raises(ValueError):
            ohet._prepare_data(data)

    def test__prepare_data_nested_lists(self):
        # Setup
        ohet = OneHotEncodingTransformer()
        data = [[[]]]

        # Assert
        with pytest.raises(ValueError):
            ohet._prepare_data(data)

    def test__prepare_data_list_of_lists(self):
        # Setup
        ohet = OneHotEncodingTransformer()

        # Run
        data = [['a'], ['b'], ['c']]
        out = ohet._prepare_data(data)

        # Assert
        expected = np.array(['a', 'b', 'c'])
        np.testing.assert_array_equal(out, expected)

    def test__prepare_data_pandas_series(self):
        # Setup
        ohet = OneHotEncodingTransformer()

        # Run
        data = pd.Series(['a', 'b', 'c'])
        out = ohet._prepare_data(data)

        # Assert
        expected = pd.Series(['a', 'b', 'c'])
        np.testing.assert_array_equal(out, expected)

    def test_fit_no_nans(self):
        # Setup
        ohet = OneHotEncodingTransformer()

        # Run
        data = pd.Series(['a', 'b', 'c'])
        ohet.fit(data)

        # Assert
        np.testing.assert_array_equal(ohet.dummies, ['a', 'b', 'c'])

    def test_fit_nans(self):
        # Setup
        ohet = OneHotEncodingTransformer()

        # Run
        data = pd.Series(['a', 'b', None])
        ohet.fit(data)

        # Assert
        np.testing.assert_array_equal(ohet.dummies, ['a', 'b', np.nan])

    def test_fit_single(self):
        # Setup
        ohet = OneHotEncodingTransformer()

        # Run
        data = pd.Series(['a', 'a', 'a'])
        ohet.fit(data)

        # Assert
        np.testing.assert_array_equal(ohet.dummies, ['a'])

    def test_transform_no_nans(self):
        # Setup
        ohet = OneHotEncodingTransformer()
        data = pd.Series(['a', 'b', 'c'])
        ohet.fit(data)

        # Run
        out = ohet.transform(data)

        # Assert
        expected = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        np.testing.assert_array_equal(out, expected)

    def test_transform_nans(self):
        # Setup
        ohet = OneHotEncodingTransformer()
        data = pd.Series(['a', 'b', None])
        ohet.fit(data)

        # Run
        out = ohet.transform(data)

        # Assert
        expected = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        np.testing.assert_array_equal(out, expected)

    def test_transform_single(self):
        # Setup
        ohet = OneHotEncodingTransformer()
        data = pd.Series(['a', 'a', 'a'])
        ohet.fit(data)

        # Run
        out = ohet.transform(data)

        # Assert
        expected = np.array([
            [1],
            [1],
            [1]
        ])
        np.testing.assert_array_equal(out, expected)

    def test_transform_all_zeros(self):
        # Setup
        ohet = OneHotEncodingTransformer()
        data = pd.Series(['a'])
        ohet.fit(data)

        # Assert
        with np.testing.assert_raises(ValueError):
            ohet.transform(['b'])

    def test_reverse_transform_no_nans(self):
        # Setup
        ohet = OneHotEncodingTransformer()
        data = pd.Series(['a', 'b', 'c'])
        ohet.fit(data)

        # Run
        transformed = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        out = ohet.reverse_transform(transformed)

        # Assert
        expected = pd.Series(['a', 'b', 'c'])
        pd.testing.assert_series_equal(out, expected)

    def test_reverse_transform_nans(self):
        # Setup
        ohet = OneHotEncodingTransformer()
        data = pd.Series(['a', 'b', None])
        ohet.fit(data)

        # Run
        transformed = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        out = ohet.reverse_transform(transformed)

        # Assert
        expected = pd.Series(['a', 'b', None])
        pd.testing.assert_series_equal(out, expected)

    def test_reverse_transform_single(self):
        # Setup
        ohet = OneHotEncodingTransformer()
        data = pd.Series(['a', 'a', 'a'])
        ohet.fit(data)

        # Run
        transformed = np.array([
            [1],
            [1],
            [1]
        ])
        out = ohet.reverse_transform(transformed)

        # Assert
        expected = pd.Series(['a', 'a', 'a'])
        pd.testing.assert_series_equal(out, expected)

    def test_reverse_transform_1d(self):
        # Setup
        ohet = OneHotEncodingTransformer()
        data = pd.Series(['a', 'a', 'a'])
        ohet.fit(data)

        # Run
        transformed = np.array([1, 1, 1])
        out = ohet.reverse_transform(transformed)

        # Assert
        expected = pd.Series(['a', 'a', 'a'])
        pd.testing.assert_series_equal(out, expected)
