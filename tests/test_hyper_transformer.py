from unittest import TestCase
from unittest.mock import Mock

import numpy as np
import pandas as pd

from rdt import HyperTransformer
from rdt.transformers import (
    BaseTransformer, BooleanTransformer, CategoricalTransformer, DatetimeTransformer,
    NullTransformer, NumericalTransformer)


class TestHyperTransformerTransformer(TestCase):

    def test_get_class_base(self):
        """Test get BaseTransformer class by name"""
        # Run
        result = HyperTransformer.get_class('BaseTransformer')

        # Asserts
        self.assertEqual(
            result,
            BaseTransformer
        )

    def test_get_class_boolean(self):
        """Test get BooleanTransformer class by name"""
        # Run
        result = HyperTransformer.get_class('BooleanTransformer')

        # Asserts
        self.assertEqual(
            result,
            BooleanTransformer
        )

    def test_get_class_categorical(self):
        """Test get CategoricalTransformer class by name"""
        # Run
        result = HyperTransformer.get_class('CategoricalTransformer')

        # Asserts
        self.assertEqual(
            result,
            CategoricalTransformer
        )

    def test_get_class_datetime(self):
        """Test get DatetimeTransformer class by name"""
        # Run
        result = HyperTransformer.get_class('DatetimeTransformer')

        # Asserts
        self.assertEqual(
            result,
            DatetimeTransformer
        )

    def test_get_class_null(self):
        """Test get NullTransformer class by name"""
        # Run
        result = HyperTransformer.get_class('NullTransformer')

        # Asserts
        self.assertEqual(
            result,
            NullTransformer
        )

    def test_get_class_numerical(self):
        """Test get NumericalTransformer class by name"""
        # Run
        result = HyperTransformer.get_class('NumericalTransformer')

        # Asserts
        self.assertEqual(
            result,
            NumericalTransformer
        )

    def test__load_transformer_basetransformer_instance(self):
        """Test _load_transformer with a basetransformer instance"""
        # Run
        transformer = BaseTransformer()

        HyperTransformer._load_transformer(transformer)

        # Asserts

    def test__load_transformer_dict_str_boolean(self):
        """Test _load_transformer with a str, BooleanTransformer"""
        # Run
        transformer = {
            'class': 'BooleanTransformer',
            'kwargs': {
                'nan': 0,
                'null_column': True
            }
        }

        result = HyperTransformer._load_transformer(transformer)

        # Asserts
        self.assertIsInstance(result, BooleanTransformer)
        self.assertEqual(result.nan, 0)
        self.assertEqual(result.null_column, True)

    def test__load_transformer_dict_str_categorical(self):
        """Test _load_transformer with a str, CategoricalTransformer"""
        # Run
        transformer = {
            'class': 'CategoricalTransformer',
            'kwargs': {
                'anonymize': 'email'
            }
        }

        result = HyperTransformer._load_transformer(transformer)

        # Asserts
        self.assertIsInstance(result, CategoricalTransformer)
        self.assertEqual(result.anonymize, 'email')

    def test__load_transformer_dict_str_datetime(self):
        """Test _load_transformer with a str, DatetimeTransformer"""
        # Run
        transformer = {
            'class': 'DatetimeTransformer',
            'kwargs': {
                'nan': 'ignore',
                'null_column': True
            }
        }

        result = HyperTransformer._load_transformer(transformer)

        # Asserts
        self.assertIsInstance(result, DatetimeTransformer)
        self.assertEqual(result.nan, 'ignore')
        self.assertEqual(result.null_column, True)

    def test__load_transformer_dict_str_null(self):
        """Test _load_transformer with a str, NullTransformer"""
        # Run
        transformer = {
            'class': 'NullTransformer',
            'kwargs': {
                'fill_value': 0,
                'null_column': True,
                'copy': True
            }
        }

        result = HyperTransformer._load_transformer(transformer)

        # Asserts
        self.assertIsInstance(result, NullTransformer)
        self.assertEqual(result.fill_value, 0)
        self.assertEqual(result.null_column, True)
        self.assertEqual(result.copy, True)

    def test__load_transformer_dict_str_numerical(self):
        """Test _load_transformer with a str, NumericalTransformer"""
        # Run
        transformer = {
            'class': 'NumericalTransformer',
            'kwargs': {
                'dtype': float,
                'nan': 'ignore',
                'null_column': True
            }
        }

        result = HyperTransformer._load_transformer(transformer)

        # Asserts
        self.assertIsInstance(result, NumericalTransformer)
        self.assertEqual(result.dtype, float)
        self.assertEqual(result.nan, 'ignore')
        self.assertEqual(result.null_column, True)

    def test___init__(self):
        """Test create new instance of HyperTransformer"""
        # Run
        ht = HyperTransformer()

        # Asserts
        self.assertEqual(ht.copy, True)
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

    def test__analyze_raise_error(self):
        """Test _analyze raise error"""
        # Setup
        data = pd.DataFrame({
            'foo': [0, 1.1, None, True, 'bar']
        })

        dtypes = [complex]

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
