from unittest import TestCase
from unittest.mock import Mock, patch

import copulas
import numpy as np
import pandas as pd
import pytest

from rdt.transformers.numerical import GaussianCopulaTransformer, NumericalTransformer


class TestNumericalTransformer(TestCase):

    def test___init__(self):
        """Test default instance"""
        # Run
        transformer = NumericalTransformer()

        # Asserts
        self.assertEqual(transformer.nan, 'mean', "Unexpected nan")
        self.assertIsNone(transformer.null_column, "null_column is None by default")
        self.assertIsNone(transformer.dtype, "dtype is None by default")

    def test_fit(self):
        """Test fit nan mean with numpy.array"""
        # Setup
        data = np.array([1.5, None, 2.5])

        # Run
        transformer = NumericalTransformer(dtype=np.float, nan='nan')
        transformer.fit(data)

        # Asserts
        expect_fill_value = 'nan'
        expect_dtype = np.float

        self.assertEqual(
            transformer.null_transformer.fill_value,
            expect_fill_value,
            "Data mean is wrong"
        )

        self.assertEqual(
            transformer._dtype,
            expect_dtype,
            "Expected dtype: float"
        )

    def test_transform_array(self):
        """Test transform numpy.array"""
        # Setup
        data = np.array([1.5, None, 2.5])

        # Run
        transformer = Mock()
        NumericalTransformer.transform(transformer, data)

        # Asserts
        expect_call_count = 1

        self.assertEqual(
            transformer.null_transformer.transform.call_count,
            expect_call_count,
            "Transform must be called only once"
        )

    def test_transform_series(self):
        """Test transform pandas.Series"""
        # Setup
        data = pd.Series([1.5, None, 2.5])

        # Run
        transformer = Mock()
        NumericalTransformer.transform(transformer, data)

        # Asserts
        expect_call_count = 1

        self.assertEqual(
            transformer.null_transformer.transform.call_count,
            expect_call_count,
            "Transform must be called only once"
        )

    def test_reverse_transform_nan_ignore(self):
        """Test reverse_transform with nan equal to ignore"""
        # Setup
        data = pd.Series([1.5, None, 2.5])

        # Run
        transformer = Mock()
        transformer.nan = None
        transformer._dtype = np.float

        result = NumericalTransformer.reverse_transform(transformer, data)

        # Asserts
        expect = pd.Series([1.5, None, 2.5])
        expected_reverse_transform_call_count = 0

        pd.testing.assert_series_equal(result, expect)
        self.assertEqual(
            transformer.null_transformer.reverse_transform.call_count,
            expected_reverse_transform_call_count,
            "NullTransformer.reverse_transform can't be called when nan is ignore"
        )

    def test_reverse_transform_nan_not_ignore(self):
        """Test reverse_transform with nan not equal to ignore"""
        # Setup
        data = pd.Series([1.5, 2.0, 2.5])
        reversed_data = pd.Series([1.5, 2.0, 2.5])

        # Run
        transformer = Mock()
        transformer.nan = 'mean'
        transformer._dtype = np.float
        transformer.null_transformer.nulls = False
        transformer.null_transformer.reverse_transform.return_value = reversed_data

        NumericalTransformer.reverse_transform(transformer, data)

        # Asserts
        expected_reverse_transform_call_count = 1

        self.assertEqual(
            transformer.null_transformer.reverse_transform.call_count,
            expected_reverse_transform_call_count,
            "NullTransformer.reverse_transform must be called at least once"
        )

    @patch('numpy.round')
    def test_reverse_transform_dtype_int(self, numpy_mock):
        """Test reverse_transform with dtype equal to int"""
        # Setup
        numpy_mock.return_value = pd.Series([3, 2, 3])
        data = pd.Series([3.0, 2.0, 3.0])

        # Run
        transformer = Mock()
        transformer.nan = None
        transformer._dtype = np.int

        result = NumericalTransformer.reverse_transform(transformer, data)

        # Asserts
        expect = pd.Series([3, 2, 3])
        expected_reverse_transform_call_count = 0

        pd.testing.assert_series_equal(result, expect)
        self.assertEqual(
            transformer.null_transformer.reverse_transform.call_count,
            expected_reverse_transform_call_count,
            "NullTransformer.reverse_transform must be called at least once"
        )


class TestGaussianCopulaTransformer:

    def test___init__super_attrs(self):
        """super() arguments are properly passed and set as attributes."""
        ct = GaussianCopulaTransformer(dtype='int', nan='mode', null_column=False)

        assert ct.dtype == 'int'
        assert ct.nan == 'mode'
        assert ct.null_column is False

    def test___init__str_distr(self):
        """If distribution is an str, it is resolved using the _DISTRIBUTIONS dict."""
        ct = GaussianCopulaTransformer(distribution='univariate')

        assert ct._distribution is copulas.univariate.Univariate

    def test___init__non_distr(self):
        """If distribution is not an str, it is store as given."""
        univariate = copulas.univariate.Univariate()
        ct = GaussianCopulaTransformer(distribution=univariate)

        assert ct._distribution is univariate

    def test__get_univariate_instance(self):
        """If a univariate instance is passed, make a copy."""
        distribution = copulas.univariate.Univariate()
        ct = GaussianCopulaTransformer(distribution=distribution)

        univariate = ct._get_univariate()

        assert univariate is not distribution
        assert isinstance(univariate, copulas.univariate.Univariate)
        assert dir(univariate) == dir(distribution)

    def test__get_univariate_tuple(self):
        """If a tuple is passed, create an instance using the given args."""
        distribution = (
            copulas.univariate.Univariate,
            {'candidates': 'a_candidates_list'}
        )
        ct = GaussianCopulaTransformer(distribution=distribution)

        univariate = ct._get_univariate()

        assert isinstance(univariate, copulas.univariate.Univariate)
        assert univariate.candidates == 'a_candidates_list'

    def test__get_univariate_class(self):
        """If a class is passed, create an instance without args."""
        distribution = copulas.univariate.Univariate
        ct = GaussianCopulaTransformer(distribution=distribution)

        univariate = ct._get_univariate()

        assert isinstance(univariate, copulas.univariate.Univariate)

    def test__get_univariate_error(self):
        """If something else is passed, rasie a TypeError."""
        distribution = 123
        ct = GaussianCopulaTransformer(distribution=distribution)

        with pytest.raises(TypeError):
            ct._get_univariate()
