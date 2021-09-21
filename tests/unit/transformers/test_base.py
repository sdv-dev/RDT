from unittest import TestCase

from rdt.transformers import BaseTransformer

import pandas as pd


class TestBaseTransformer(TestCase):

    def setUp(self):
        self.data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': ['x', 'y', 'z']
        })
        self.base = BaseTransformer()

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