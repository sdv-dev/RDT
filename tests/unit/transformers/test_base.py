from unittest import TestCase
from unittest.mock import MagicMock

import pandas as pd

from rdt.transformers import BaseTransformer, CategoricalTransformer


class TestBaseTransformer(TestCase):

    def setUp(self):
        self.data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': ['x', 'y', 'z']
        })
        self.base = BaseTransformer()

    def test_get_input_type(self):
        """Test `get_input_type` returns `_INPUT_TYPE` of child class."""
        categorical_transformer = CategoricalTransformer()
        categorical_transformer.INPUT_TYPE = MagicMock('categorical')
        self.assertEqual(categorical_transformer.get_input_type(), 'categorical')

    def test__add_prefix_none(self):
        """Test `_add_prefix`."""
        self.assertEqual(self.base._add_prefix(None, None), None)

    def test__add_prefix(self):
        """Test `_add_prefix`."""
        column_to_type = {'digit': 'numerical', 'letter': 'categorical'}
        expected = {'prefix.digit': 'numerical', 'prefix.letter': 'categorical'}
        self.assertEqual(self.base._add_prefix(column_to_type, 'prefix'), expected)

    def test_get_output_types(self):
        """Test `get_output_types`."""
        categorical_transformer = CategoricalTransformer()
        categorical_transformer.OUTPUT_TYPES = MagicMock(
            {'value': 'categorical', 'is_null': 'null'})
        categorical_transformer._column_prefix = MagicMock('prefix')
        expected = {'prefix.value': 'categorical', 'prefix.is_null': 'null'}
        self.assertEqual(categorical_transformer.get_output_types(), expected)

    def test_is_transform_deterministic(self):
        """Test `is_transform_deterministic`."""
        categorical_transformer = CategoricalTransformer()
        categorical_transformer.DETERMINISTIC_TRANSFORM = MagicMock(False)
        self.assertEqual(categorical_transformer.is_transform_deterministic(), False)

    def test__fit(self):
        """Test call fit and pass"""
        with self.assertRaises(NotImplementedError):
            self.base._fit(None)

    def test__transform(self):
        """Test call transform raise not implemented error"""
        with self.assertRaises(NotImplementedError):
            self.base._transform(None)

    def test_fit_transform(self):
        """Test call fit_transform raise not implemented error"""
        with self.assertRaises(NotImplementedError):
            self.base.fit_transform(self.data, ['a'])

    def test__reverse_transform(self):
        """Test call reverse_transform raise not implemented error"""
        with self.assertRaises(NotImplementedError):
            self.base._reverse_transform(None)

    def test_fit1(self):
        """Test the fit method."""
        with self.assertRaises(NotImplementedError):
            self.base.fit(self.data, ['a'])

        assert self.base._column_prefix == 'a'

    def test_fit2(self):
        """Test the fit method."""
        with self.assertRaises(NotImplementedError):
            self.base.fit(self.data, ['a', 'b'])

        assert self.base._column_prefix == 'a#b'
