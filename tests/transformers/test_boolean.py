from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from rdt.transformers import BooleanTransformer


class TestBooleanTransformer(TestCase):

    def test___init__(self):
        """Test default instance"""
        # Run
        transformer = BooleanTransformer()

        # Asserts
        self.assertEqual(transformer.nan, -1, "Unexpected nan")
        self.assertIsNone(transformer.null_column, "null_column is None by default")

    def test_fit_nan_ignore(self):
        """Test fit nan equal to ignore"""
        # Setup
        data = pd.Series([False, True, True, False, True])

        # Run
        transformer = BooleanTransformer(nan='ignore')
        transformer.fit(data)

        # Asserts
        expect_fill_value = None

        self.assertEqual(
            transformer.null_transformer.fill_value,
            expect_fill_value,
            "Unexpected fill value"
        )

    def test_fit_nan_not_ignore(self):
        """Test fit nan not equal to ignore"""
        # Setup
        data = pd.Series([False, True, True, False, True])

        # Run
        transformer = BooleanTransformer(nan=0)
        transformer.fit(data)

        # Asserts
        expect_fill_value = 0

        self.assertEqual(
            transformer.null_transformer.fill_value,
            expect_fill_value,
            "Unexpected fill value"
        )
