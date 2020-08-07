from unittest import TestCase

from rdt.transformers import BaseTransformer


class TestNumericalTransformer(TestCase):

    def setUp(self):
        self.base = BaseTransformer()

    def test_fit(self):
        """Test call fit and pass"""
        with self.assertRaises(NotImplementedError):
            self.base.fit(None)

    def test_transform(self):
        """Test call transform raise not implemented error"""
        with self.assertRaises(NotImplementedError):
            self.base.transform(None)

    def test_fit_transform(self):
        """Test call fit_transform raise not implemented error"""
        with self.assertRaises(NotImplementedError):
            self.base.fit_transform(None)

    def test_reverse_transform(self):
        """Test call reverse_transform raise not implemented error"""
        with self.assertRaises(NotImplementedError):
            self.base.reverse_transform(None)
