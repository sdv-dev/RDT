from unittest import TestCase
from unittest.mock import Mock, call, patch

import numpy as np
import pandas as pd

from rdt.transformers import DateTimeTransformer


class TestDTTransformer(TestCase):

    def test___init__(self):
        # Run
        transformer = DateTimeTransformer('%Y-%m-%d')

        # Asserts
        assert transformer.datetime_format == '%Y-%m-%d'

    def test_transform_series(self):
        """Transform a pandas.Series."""
        # Setup

        # Run
        transformer = Mock()
        transformer.datetime_format = '%Y-%m-%d'

        data = pd.Series(['1996-10-17', None, '1965-05-23'])

        result = DateTimeTransformer.transform(transformer, data)

        # Asserts
        expect = np.array([845510400000000000, -145497600000000000])

        assert np.array_equal(result, expect)

    def test_transform_array(self):
        """Transform a numpy.array."""
        # Setup

        # Run
        transformer = Mock()
        transformer.datetime_format = '%Y-%m-%d'

        data = np.array(['1996-10-17', None, '1965-05-23'])

        result = DateTimeTransformer.transform(transformer, data)

        # Asserts
        expect = np.array([845510400000000000, -145497600000000000])

        assert np.array_equal(result, expect)

    def test__transform_to_date_epoch(self):
        """Transform 0 into epoch time"""
        # Setup

        # Run
        transformer = Mock()
        transformer.datetime_format = '%Y-%m-%d'

        data = '0'

        result = DateTimeTransformer._transform_to_date(transformer, data)

        # Asserts
        expect = '1970-01-01'

        assert result == expect

    def test__transform_to_date_pre_epoch(self):
        """Transform a negative value into pre epoch time"""
        # Setup

        # Run
        transformer = Mock()
        transformer.datetime_format = '%Y-%m-%d'

        data = '-145497600000000000'

        result = DateTimeTransformer._transform_to_date(transformer, data)

        # Asserts
        expect = '1965-05-23'

        assert result == expect

    def test__transform_to_date_post_epoch(self):
        """Transform positive value into post epoch time"""
        # Setup

        # Run
        transformer = Mock()
        transformer.datetime_format = '%Y-%m-%d'

        data = '845510400000000000'

        result = DateTimeTransformer._transform_to_date(transformer, data)

        # Asserts
        expect = '1996-10-17'

        assert result == expect

    @patch('numpy.vectorize')
    def test_reverse_transform(self, np_vectorize_mock):
        # Setup

        # Run
        transformer = Mock()
        transformer._transform_to_date = None

        data = np.array([845510400000000000, -145497600000000000])

        DateTimeTransformer.reverse_transform(transformer, data)

        # Asserts
        assert np_vectorize_mock.call_count == 1
        assert np_vectorize_mock.call_args == call(None)
