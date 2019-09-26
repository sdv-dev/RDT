from unittest import TestCase
from unittest.mock import Mock, call, patch

import numpy as np
import pandas as pd

from rdt.transformers import NumericalTransformer


class TestNumericalTransformer(TestCase):

    def test_fit_pandas_series(self):
        """Save dtype and compute the mean with a pandas.Series."""
        # Setup

        # Run
        transformer = Mock()

        data = pd.Series([1.5, None, 2.5])

        NumericalTransformer.fit(transformer, data)

        # Asserts
        transformer.dtype == 'float64'
        transformer.default == 2.0

    def test_fit_numpy_array(self):
        """Save dtype and compute the mean with a numpy.array."""
        # Setup

        # Run
        transformer = Mock()

        data = np.array([1.5, None, 2.5])

        NumericalTransformer.fit(transformer, data)

        # Asserts
        transformer.dtype == 'float64'
        transformer.default == 2.0

    def test_transform_pandas_series(self):
        """Transform pandas.Series."""
        # Setup

        # Run
        transformer = Mock()
        transformer.default = 2.0

        data = pd.Series([1.5, None, 2.5])

        result = NumericalTransformer.transform(transformer, data)

        # Asserts
        expect = np.array([1.5, 2.0, 2.5])

        assert np.array_equal(result, expect)
        assert isinstance(result, np.ndarray)

    def test_transform_numpy_array(self):
        """Transform numpy.array."""
        # Setup

        # Run
        transformer = Mock()
        transformer.default = 2.0

        data = np.array([1.5, None, 2.5])

        result = NumericalTransformer.transform(transformer, data)

        # Asserts
        expect = np.array([1.5, 2.0, 2.5])

        assert np.array_equal(result, expect)
        assert isinstance(result, np.ndarray)

    def test_reverse_transform(self):
        """Reverse transformed data."""
        # Setup

        # Run
        transformer = Mock()
        transformer.dtype = 'float64'

        data = np.array([1.5, 2.0, 2.5])

        result = NumericalTransformer.reverse_transform(transformer, data)

        # Asserts
        expect = np.array([1.5, 2.0, 2.5])

        assert np.array_equal(result, expect)

    def test_reverse_transform_error(self):
        """Raises an error in reverse_transform."""
        # Setup

        # Run and aasserts
        transformer = Mock()
        transformer.dtype = 'float64'

        data = None
        
        with self.assertRaises(ValueError):
            NumericalTransformer.reverse_transform(transformer, data)
