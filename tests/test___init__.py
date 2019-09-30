from unittest import TestCase
from unittest.mock import patch

import pandas as pd

from rdt import _lookup, load_data


class TestLookup(TestCase):

    def test__lookup_ok(self):
        """Test _lookup return element."""
        # Setup

        # Run
        elements = [{'foo': 0}]
        field = 'foo'
        value = 0

        result = _lookup(elements, field, value)

        # Asserts
        expect = {'foo': 0}
        assert result == expect

    def test__lookup_error(self):
        """Test _lookup raise error."""
        # Setup

        # Run
        elements = [{'foo': 0}]
        field = 'foo'
        value = 1

        with self.assertRaises(ValueError):
            _lookup(elements, field, value)


class TestLoadData(TestCase):

    @patch('rdt._lookup')
    @patch('pandas.read_csv')
    @patch('json.load')
    @patch('builtins.open')
    def test_load_column(self, open_mock, load_mock, pandas_mock, lookup_mock):
        """Load single column."""

        # Setup
        load_mock.return_value = {
            'path': '',
            'tables': [{
                'name': 'foo',
                'path': 'foo.csv'
            }]
        }

        pandas_mock.return_value = pd.DataFrame({'foo': [0, 1]})

        lookup_mock.side_effect = [
            {'path': '', 'fields': []},
            {'col': 'meta'}
        ]

        # Run
        metadata_path = 'some_path'
        table_name = 'foo'

        load_data(metadata_path, table_name)

        # Asserts
        assert lookup_mock.call_count == 1

    @patch('rdt._lookup')
    @patch('pandas.read_csv')
    @patch('json.load')
    @patch('builtins.open')
    def test_load_table(self, open_mock, load_mock, pandas_mock, lookup_mock):
        """Load table."""

        # Setup
        load_mock.return_value = {
            'path': '',
            'tables': [{
                'name': 'foo',
                'path': 'foo.csv',
                'fields': [{
                    'name': 'a_field'
                }]
            }]
        }

        pandas_mock.return_value = pd.DataFrame({'a_field': [0, 1]})

        lookup_mock.side_effect = [
            {'path': '', 'fields': []},
            {'col': 'meta'}
        ]

        # Run
        metadata_path = 'some_path'
        table_name = 'foo'
        column_name = 'a_field'

        load_data(metadata_path, table_name, column_name)

        # Asserts
        assert lookup_mock.call_count == 2
