from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from rdt import HyperTransformer
from rdt.transformers import (
    BooleanTransformer, CategoricalTransformer, DatetimeTransformer, NumericalTransformer)


class TestHyperTransformerTransformer(TestCase):

    def test___init__(self):
        """Test create new instance of HyperTransformer"""
        # Run
        ht = HyperTransformer()

        # Asserts
        self.assertTrue(ht.copy)
        self.assertEqual(ht.anonymize, dict())
        self.assertEqual(ht.dtypes, None)

    def test__analyze_int(self):
        """Test _analyze int dtype"""
        # Setup
        data = pd.DataFrame({
            'integers': [1, 2, 3, 4, 5, None, 6, 7, 8, 9, 0]
        })

        dtypes = [int]

        # Run
        transformer = Mock()
        transformer.dtypes = dtypes

        result = HyperTransformer._analyze(transformer, data)

        # Asserts
        expect_class = NumericalTransformer

        self.assertIsInstance(result['integers'], expect_class)

    def test__analyze_float(self):
        """Test _analyze float dtype"""
        # Setup
        data = pd.DataFrame({
            'floats': [1.1, 2.2, 3.3, 4.4, 5.5, None, 6.6, 7.7, 8.8, 9.9, 0.0]
        })

        dtypes = [float]

        # Run
        transformer = Mock()
        transformer.dtypes = dtypes

        result = HyperTransformer._analyze(transformer, data)

        # Asserts
        expect_class = NumericalTransformer

        self.assertIsInstance(result['floats'], expect_class)

    def test__analyze_object(self):
        """Test _analyze object dtype"""
        # Setup
        data = pd.DataFrame({
            'objects': ['foo', 'bar', None, 'tar']
        })

        dtypes = [np.object]

        # Run
        transformer = Mock()
        transformer.dtypes = dtypes

        result = HyperTransformer._analyze(transformer, data)

        # Asserts
        expect_class = CategoricalTransformer

        self.assertIsInstance(result['objects'], expect_class)

    def test__analyze_bool(self):
        """Test _analyze bool dtype"""
        # Setup
        data = pd.DataFrame({
            'booleans': [True, False, None, False, True]
        })

        dtypes = [bool]

        # Run
        transformer = Mock()
        transformer.dtypes = dtypes

        result = HyperTransformer._analyze(transformer, data)

        # Asserts
        expect_class = BooleanTransformer

        self.assertIsInstance(result['booleans'], expect_class)

    def test__analyze_datetime64(self):
        """Test _analyze datetime64 dtype"""
        # Setup
        data = pd.DataFrame({
            'datetimes': ['1965-05-23', None, '1997-10-17']
        })

        data['datetimes'] = pd.to_datetime(data['datetimes'], format='%Y-%m-%d', errors='coerce')

        dtypes = [np.datetime64]

        # Run
        transformer = Mock()
        transformer.dtypes = dtypes

        result = HyperTransformer._analyze(transformer, data)

        # Asserts
        expect_class = DatetimeTransformer

        self.assertIsInstance(result['datetimes'], expect_class)

    @patch('rdt.hyper_transformer.np.dtype', new=Mock())
    def test__analyze_raise_error(self):
        """Test _analyze raise error"""
        # Setup
        data = Mock()
        data.columns = ['foo']

        dtypes = [Mock()]

        # Run
        transformer = Mock()
        transformer.dtypes = dtypes

        with self.assertRaises(ValueError):
            HyperTransformer._analyze(transformer, data)

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
        })

        returned = HyperTransformer._get_columns(data, 'a')

        np.testing.assert_equal(returned, np.array([1, 2, 3]))

    def test__get_columns_two(self):
        data = pd.DataFrame({
            'b': [4, 5, 6],
            'b#1': [7, 8, 9],
        })

        returned = HyperTransformer._get_columns(data, 'b')

        expected = np.array([
            [4, 7],
            [5, 8],
            [6, 9]
        ])
        np.testing.assert_equal(returned, expected)

    def test__get_columns_error(self):
        data = pd.DataFrame({
            'a': [1, 2, 3],
        })

        with pytest.raises(ValueError):
            HyperTransformer._get_columns(data, 'b')

    def test__get_columns_regex(self):
        data = pd.DataFrame({
            'a(b)': [4, 5, 6],
            'a(b)#1': [7, 8, 9],
        })

        returned = HyperTransformer._get_columns(data, 'a(b)')

        expected = np.array([
            [4, 7],
            [5, 8],
            [6, 9]
        ])
        np.testing.assert_equal(returned, expected)
