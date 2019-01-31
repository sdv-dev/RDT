import json
import os
from unittest import TestCase, skip
from unittest.mock import patch

import pandas as pd

from rdt.hyper_transformer import HyperTransformer
from tests import safe_compare_dataframes


class TestHyperTransformer(TestCase):

    def test___init__(self):
        """On init, meta file is the only required argument, other attributes are setup."""
        # Setup
        expected_transformer_dict = {
            ('users', 'id'): 'categorical',
            ('users', 'date_account_created'): 'datetime',
            ('users', 'timestamp_first_active'): 'datetime',
            ('users', 'date_first_booking'): 'datetime',
            ('users', 'gender'): 'categorical',
            ('users', 'age'): 'number',
            ('users', 'signup_method'): 'categorical',
            ('users', 'signup_flow'): 'categorical',
            ('users', 'language'): 'categorical',
            ('users', 'affiliate_channel'): 'categorical',
            ('users', 'affiliate_provider'): 'categorical',
            ('users', 'first_affiliate_tracked'): 'categorical',
            ('users', 'signup_app'): 'categorical',
            ('users', 'first_device_type'): 'categorical',
            ('users', 'first_browser'): 'categorical',
            ('countries', 'country_destination'): 'categorical',
            ('countries', 'lat_destination'): 'number',
            ('countries', 'lng_destination'): 'number',
            ('countries', 'distance_km'): 'number',
            ('countries', 'destination_km2'): 'categorical',
            ('countries', 'destination_language '): 'categorical',
            ('countries', 'language_levenshtein_distance'): 'number',
            ('sessions', 'user_id'): 'categorical',
            ('sessions', 'action'): 'categorical',
            ('sessions', 'action_type'): 'categorical',
            ('sessions', 'action_detail'): 'categorical',
            ('sessions', 'device_type'): 'categorical',
            ('sessions', 'secs_elapsed'): 'number',
            ('age_gender_bkts', 'age_bucket'): 'categorical',
            ('age_gender_bkts', 'country_destination'): 'categorical',
            ('age_gender_bkts', 'gender'): 'categorical',
            ('age_gender_bkts', 'population_in_thousands'): 'number',
            ('age_gender_bkts', 'year'): 'datetime'
        }

        # Run
        ht = HyperTransformer('tests/data/airbnb/airbnb_meta.json')

        # Check
        assert set(ht.table_dict.keys()) == {'users', 'sessions'}
        assert ht.transformer_dict == expected_transformer_dict

    def test___init__metadata_dict(self):
        """On init, meta file is the only required argument, other attributes are setup."""
        # Setup
        expected_transformer_dict = {
            ('users', 'id'): 'categorical',
            ('users', 'date_account_created'): 'datetime',
            ('users', 'timestamp_first_active'): 'datetime',
            ('users', 'date_first_booking'): 'datetime',
            ('users', 'gender'): 'categorical',
            ('users', 'age'): 'number',
            ('users', 'signup_method'): 'categorical',
            ('users', 'signup_flow'): 'categorical',
            ('users', 'language'): 'categorical',
            ('users', 'affiliate_channel'): 'categorical',
            ('users', 'affiliate_provider'): 'categorical',
            ('users', 'first_affiliate_tracked'): 'categorical',
            ('users', 'signup_app'): 'categorical',
            ('users', 'first_device_type'): 'categorical',
            ('users', 'first_browser'): 'categorical',
            ('countries', 'country_destination'): 'categorical',
            ('countries', 'lat_destination'): 'number',
            ('countries', 'lng_destination'): 'number',
            ('countries', 'distance_km'): 'number',
            ('countries', 'destination_km2'): 'categorical',
            ('countries', 'destination_language '): 'categorical',
            ('countries', 'language_levenshtein_distance'): 'number',
            ('sessions', 'user_id'): 'categorical',
            ('sessions', 'action'): 'categorical',
            ('sessions', 'action_type'): 'categorical',
            ('sessions', 'action_detail'): 'categorical',
            ('sessions', 'device_type'): 'categorical',
            ('sessions', 'secs_elapsed'): 'number',
            ('age_gender_bkts', 'age_bucket'): 'categorical',
            ('age_gender_bkts', 'country_destination'): 'categorical',
            ('age_gender_bkts', 'gender'): 'categorical',
            ('age_gender_bkts', 'population_in_thousands'): 'number',
            ('age_gender_bkts', 'year'): 'datetime'
        }

        # Run
        path = 'tests/data/airbnb/airbnb_meta.json'
        dir_name = os.path.dirname(path)
        with open(path, 'r') as f:
            metadata = json.load(f)
        ht = HyperTransformer(metadata, dir_name)

        # Check
        assert set(ht.table_dict.keys()) == {'users', 'sessions'}
        assert ht.transformer_dict == expected_transformer_dict

    def test_get_class(self):
        """Get a transformer from its name."""

        # Setup
        ht = HyperTransformer('tests/data/airbnb/airbnb_meta.json')

        # Run
        transformer = ht.get_class('DTTransformer')

        # Check
        assert transformer.__name__ == 'DTTransformer'

    @patch('rdt.hyper_transformer.pd.read_csv')
    def test__get_tables(self, read_mock):
        """_get_tables loads the data from disk and structures it along its metadata."""

        # Setup
        metadata = {
            'path': '',
            'tables': [{
                'fields': [
                    {
                        'name': 'A',
                        'type': 'numerical',
                    },
                    {
                        'name': 'B',
                        'type': 'numerical',
                    },
                    {
                        'name': 'primary_key',
                        'type': 'numerical',
                    },
                ],
                'headers': True,
                'name': 'table_name',
                'path': 'table.csv',
                'primary_key': 'primary_key',
                'use': True
            }]
        }
        dir_name = ''
        hyper_transformer = HyperTransformer(metadata, dir_name)

        read_mock.return_value = pd.DataFrame({
            'primary_key': [1, 2],
            'A': [1, 0],
            'B': [0, 1]
        })

        # We reset the mock because it's called on init
        read_mock.reset_mock()

        # We expect the data_table to be the read_mock unchanged because there are
        # no fields with the pii key and the data should remain unchanged.
        expected_result = {
            'table_name': (read_mock.return_value, metadata['tables'][0])
        }

        # Run
        result = hyper_transformer._get_tables(dir_name)

        # Check
        assert isinstance(result, dict)
        assert len(result.keys()) == 1

        # We check that both the dataframe and the metadata are the expected.
        actual_table = result['table_name']
        expected_table = expected_result['table_name']

        actual_data = actual_table[0]
        expected_data = expected_table[0]
        assert actual_data.equals(expected_data)

        actual_meta = actual_table[1]
        expected_meta = expected_table[1]
        assert actual_meta == expected_meta

        # We check the mock has only been called once and with the expected path.
        expected_path = 'table.csv'
        read_mock.assert_called_once_with(expected_path)

    @patch('rdt.hyper_transformer.transformers.CatTransformer')
    @patch('rdt.hyper_transformer.pd.read_csv')
    def test__get_tables_pii_fields(self, read_mock, transformer_mock):
        """If some fields have the keyword pii, they are transformed to anonymize the data."""
        # Setup
        metadata = {
            'path': '',
            'tables': [{
                'fields': [
                    {
                        'name': 'name',
                        'type': 'categorical',
                        'pii': True,
                        'pii_category': 'first_name'
                    },
                    {
                        'name': 'user_id',
                        'type': 'numerical',
                    },
                ],
                'headers': True,
                'name': 'table_name',
                'path': 'table.csv',
                'primary_key': 'user_id',
                'use': True
            }]
        }
        dir_name = ''
        hyper_transformer = HyperTransformer(metadata, dir_name)

        read_mock.return_value = pd.DataFrame({
            'name': ['Mike', 'John'],
            'user_id': [1, 2]
        })
        instance_mock = transformer_mock.return_value
        instance_mock.anonymize_column.return_value = pd.DataFrame({'name': ['Peter', 'David']})

        # We reset the mocks because it's called on init
        read_mock.reset_mock()
        transformer_mock.reset_mock()
        instance_mock.reset_mock()

        expected_data_table = pd.DataFrame({
            'name': ['Peter', 'David'],  # instance_mock.fit_transform.return_value
            'user_id': [1, 2]
        })
        expected_result = {
            'table_name': (expected_data_table, metadata['tables'][0])
        }

        # Run
        result = hyper_transformer._get_tables(dir_name)

        # Check
        assert isinstance(result, dict)
        assert len(result.keys()) == 1

        # We check that both the dataframe and the metadata are the expected.
        actual_table = result['table_name']
        expected_table = expected_result['table_name']

        actual_data = actual_table[0]
        expected_data = expected_table[0]
        assert actual_data.equals(expected_data)

        actual_meta = actual_table[1]
        expected_meta = expected_table[1]
        assert actual_meta == expected_meta

        # read_mock has only been called once as expected
        expected_path = 'table.csv'
        read_mock.assert_called_once_with(expected_path)

        # transformer mock has been called only once, for field 'name'.
        expected_field = {
            'name': 'name',
            'type': 'categorical',
            'pii': True,
            'pii_category': 'first_name'
        }
        transformer_mock.assert_called_once_with(expected_field)

        # instance_mock.anonimize_column has been called once, and was passed the unmodified data.
        df = pd.DataFrame({
            'name': ['Mike', 'John'],
            'user_id': [1, 2]
        })
        expected_call_args_list = ((df,), {})
        instance_mock.anonymize_column.call_args_list == expected_call_args_list

    @skip('https://github.com/HDI-Project/RDT/issues/52')
    def test_fit_transform_table_transformer_dict(self):
        """Create and run the specified transforms in transformed_dict over the given table."""
        # Setup
        ht = HyperTransformer('tests/data/airbnb/airbnb_meta.json')
        table, table_meta = ht.table_dict['users']
        transformer_dict = {
            ('users', 'age'): 'number',
            ('users', 'date_first_booking'): 'datetime'
        }
        expected_result = pd.DataFrame(
            {
                '?date_first_booking': [1, 0, 0, 0, 1],
                'date_first_booking': [1.38879e+18, 0.0, 0.0, 0.0, 1.3886172e+18],
                '?age': [1, 0, 0, 0, 0],
                'age': [62, 62, 62, 62, 62]
            },
            columns=['?date_first_booking', 'date_first_booking', '?age', 'age']
        )

        # Run
        result = ht.fit_transform_table(table, table_meta, transformer_dict)

        # Check
        assert result.equals(expected_result)

        for key in transformer_dict:
            with self.subTest(transformer_key=key):
                transformer = ht.transformer_dict.get(key)
                transformer_type = [
                    x['type'] for x in table_meta['fields']
                    if x['name'] == key[1]
                ][0]
                assert transformer_type == transformer

    def test_fit_transform_transformer_list(self):
        """Create and run the transformers in transformer_list on the given table."""
        # Setup
        ht = HyperTransformer('tests/data/airbnb/airbnb_meta.json')
        table, table_meta = ht.table_dict['users']
        transformer_list = ['NumberTransformer']
        expected_result = pd.DataFrame(
            {
                '?age': [1, 0, 0, 0, 0],
                'age': [62, 62, 62, 62, 62]
            },
            columns=['age', '?age']
        )

        # Run
        result = ht.fit_transform_table(table, table_meta, transformer_list=transformer_list)

        # Check
        assert result.equals(expected_result)

    def test_transform_table(self):
        """transform_table transform a whole table after being fit."""
        # Setup
        ht = HyperTransformer('tests/data/airbnb/airbnb_meta.json')
        table, table_meta = ht.table_dict['users']
        transformers = ['DTTransformer', 'NumberTransformer', 'CatTransformer']
        ht.fit_transform_table(table, table_meta, transformer_list=transformers)

        # Run
        result = ht.transform_table(table, table_meta)

        # Check
        assert (result.index == table.index).all()
        for column in table.columns:
            with self.subTest(column=column):
                missing = '?' + column
                assert column in result.columns
                assert (result[column] == pd.to_numeric(result[column])).all()

                if table[column].isnull().any():
                    assert missing in result.columns
                    assert (table[column].isnull() == (result[missing] == 0)).all()

    @skip('https://github.com/HDI-Project/RDT/issues/49')
    def test_reverse_transform_table(self):
        """reverse_transform leave transformed data in its original state."""

        # Setup
        ht = HyperTransformer('tests/data/airbnb/airbnb_meta.json')
        table, table_meta = ht.table_dict['users']
        transformers = ['DTTransformer', 'NumberTransformer', 'CatTransformer']
        ht.fit_transform_table(table, table_meta, transformer_list=transformers)

        # Run
        transformed = ht.transform_table(table, table_meta)
        reverse_transformed = ht.reverse_transform_table(transformed, table_meta)

        # Check
        for column in table.columns:
            with self.subTest(column=column):
                assert (reverse_transformed[column] == table[column]).all()

    def test_fit_transform(self):
        """Create transformers for each column/table pair and apply them on input data."""
        # Setup
        ht = HyperTransformer('tests/data/airbnb/airbnb_meta.json', missing=True)

        # Run
        result = ht.fit_transform()

        # Check
        assert set(result.keys()) == {'users', 'sessions'}
        for name, table in result.items():
            values, meta = ht.table_dict[name]
            for column in values.columns:
                with self.subTest(column=column):
                    missing = '?' + column
                    meta_col = [field for field in meta['fields'] if field['name'] == column][0]
                    assert column in table.columns
                    assert (table[column] == pd.to_numeric(table[column])).all()

                    if meta_col['type'] != 'categorical' and table[column].isnull().any():
                        # This is due to the fact that CatTransformer is able to handle
                        # nulls by itself without relying in NullTransformer.
                        assert missing in table.columns
                        assert (values[column].isnull() == (table[missing] == 0)).all()

    @skip('https://github.com/HDI-Project/RDT/issues/49')
    def test_reverse_transform(self):
        """reverse_transform leave transformed data in its original state."""
        # Setup
        ht = HyperTransformer('tests/data/airbnb/airbnb_meta.json')
        transformed = ht.fit_transform()
        original_data = {name: table[0] for name, table in ht.table_dict.items()}

        # Run
        reverse_transformed = ht.reverse_transform(transformed)

        # Check
        for name, table in original_data.items():
            reversed_table = reverse_transformed[name]
            assert safe_compare_dataframes(reversed_table, table)
