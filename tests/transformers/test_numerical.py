from unittest import TestCase

import copulas
import numpy as np
import pytest

from rdt.transformers.numerical import GaussianCopulaTransformer, NumericalTransformer


class TestNumericalTransformer(TestCase):

    def test___init__super_attrs(self):
        """super() arguments are properly passed and set as attributes."""
        nt = NumericalTransformer(dtype='int', nan='mode', null_column=False)

        assert nt.dtype == 'int'
        assert nt.nan == 'mode'
        assert nt.null_column is False

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

        assert transformer.null_transformer.fill_value == expect_fill_value
        assert transformer._dtype == expect_dtype


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
