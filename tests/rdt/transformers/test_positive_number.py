from math import e
from unittest import TestCase

import pandas as pd

from rdt.transformers.positive_number import PositiveNumberTransformer


class TestPositiveNumberTransformer(TestCase):

    def test_transform(self):
        """Any value transformed will be positive."""

        # Setup
        metadata = {
            'name': 'field',
            'type': 'number'
        }
        transformer = PositiveNumberTransformer(metadata)

        data = pd.DataFrame({
            'field': [-1.0, 0, 1.0]
        })

        expected_result = pd.DataFrame({
            'field': [e**-1, 1, e]
        })

        # Run
        result = transformer.fit_transform(data)

        # Check
        assert result.equals(expected_result)

    def test_reverse_transform(self):
        """All positive values can be reverse_transformed."""

        # Setup
        metadata = {
            'name': 'field',
            'type': 'number'
        }
        transformer = PositiveNumberTransformer(metadata)

        data = pd.DataFrame({
            'field': [e**-1, 1, e]
        })

        expected_result = pd.DataFrame({
            'field': [-1.0, 0, 1.0]
        })

        # Run
        result = transformer.reverse_transform(data)

        # Check
        assert result.equals(expected_result)
