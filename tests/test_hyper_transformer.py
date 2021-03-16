from unittest import TestCase
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from rdt import HyperTransformer
from rdt.transformers import (
    BooleanTransformer, DatetimeTransformer, NumericalTransformer, OneHotEncodingTransformer)


class TestHyperTransformer(TestCase):

    def test___init__(self):
        """Test create new instance of HyperTransformer"""
        # Run
        ht = HyperTransformer()

        # Asserts
        self.assertTrue(ht.copy)
        self.assertEqual(ht.dtypes, None)

    def test__analyze(self):
        """Test _analyze"""
        # Setup
        hp = HyperTransformer(dtype_transformers={'O': 'one_hot_encoding'})

        # Run
        data = pd.DataFrame({
            'int': [1, 2, None],
            'float': [1.0, 2.0, None],
            'object': ['foo', 'bar', None],
            'category': [1, 2, None],
            'bool': [True, False, None],
            'datetime': pd.to_datetime(['1965-05-23', None, '1997-10-17']),
        })
        data['category'] = data['category'].astype('category')
        result = hp._analyze(data)

        # Asserts
        assert isinstance(result, dict)
        assert set(result.keys()) == {'int', 'float', 'object', 'category', 'bool', 'datetime'}

        assert isinstance(result['int'], NumericalTransformer)
        assert isinstance(result['float'], NumericalTransformer)
        assert isinstance(result['object'], OneHotEncodingTransformer)
        assert isinstance(result['category'], OneHotEncodingTransformer)
        assert isinstance(result['bool'], BooleanTransformer)
        assert isinstance(result['datetime'], DatetimeTransformer)

    def test__analyze_invalid_dtype(self):
        """Test _analyze when a list of dtypes containing an invalid dtype is passed."""
        # Setup
        hp = HyperTransformer(dtypes=['int', 'complex'])

        # Run
        data = pd.DataFrame({
            'int': [1, 2, None],
            'complex': [1.0 + 0j, 2.0 + 1j, None],
        })
        with pytest.raises(ValueError):
            hp._analyze(data)

    def test_fit_with_analyze(self):
        """Test fit and analyze the transformers"""
        # Setup
        data = pd.DataFrame({
            'integers': [1, 2, 3, 4],
            'floats': [1.1, 2.2, 3.3, 4.4],
            'booleans': [True, False, False, True]
        })

        int_mock = Mock()
        float_mock = Mock()
        bool_mock = Mock()

        analyzed_data = {
            'integers': int_mock,
            'floats': float_mock,
            'booleans': bool_mock
        }

        # Run
        transformer = Mock()
        transformer.transformers = None
        transformer._analyze.return_value = analyzed_data

        HyperTransformer.fit(transformer, data)

        # Asserts
        expect_int_call_count = 1
        expect_float_call_count = 1
        expect_bool_call_count = 1

        self.assertEqual(int_mock.fit.call_count, expect_int_call_count)
        self.assertEqual(float_mock.fit.call_count, expect_float_call_count)
        self.assertEqual(bool_mock.fit.call_count, expect_bool_call_count)

    def test_fit_transform(self):
        """Test call fit_transform"""
        # Run
        transformer = Mock()

        HyperTransformer.fit_transform(transformer, pd.DataFrame())

        # Asserts
        expect_call_count_fit = 1
        expect_call_count_transform = 1
        expect_call_args_fit = pd.DataFrame()
        expect_call_args_transform = pd.DataFrame()

        self.assertEqual(
            transformer.fit.call_count,
            expect_call_count_fit
        )
        pd.testing.assert_frame_equal(
            transformer.fit.call_args[0][0],
            expect_call_args_fit
        )

        self.assertEqual(
            transformer.transform.call_count,
            expect_call_count_transform
        )
        pd.testing.assert_frame_equal(
            transformer.transform.call_args[0][0],
            expect_call_args_transform
        )

    def test__get_columns_one(self):
        data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [1, 2, 3],
        })

        returned = HyperTransformer._get_columns(data, 'a')

        np.testing.assert_equal(returned, np.array(['a']))

    def test__get_columns_two(self):
        data = pd.DataFrame({
            'a': [4, 5, 6],
            'a#1': [7, 8, 9],
            'b': [4, 5, 6],
            'b#1': [7, 8, 9],
        })

        returned = HyperTransformer._get_columns(data, 'b')

        expected = np.array([
            'b',
            'b#1',
        ])
        np.testing.assert_equal(returned, expected)

    def test__get_columns_none(self):
        data = pd.DataFrame({
            'a': [1, 2, 3],
        })

        returned = HyperTransformer._get_columns(data, 'b')

        assert returned.empty

    def test__get_columns_regex(self):
        data = pd.DataFrame({
            'a(b)': [4, 5, 6],
            'a(b)#1': [7, 8, 9],
            'b(b)': [4, 5, 6],
            'b(b)#1': [7, 8, 9],
        })

        returned = HyperTransformer._get_columns(data, 'a(b)')

        expected = np.array([
            'a(b)',
            'a(b)#1',
        ])
        np.testing.assert_equal(returned, expected)
