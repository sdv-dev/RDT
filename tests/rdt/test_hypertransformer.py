from unittest import TestCase, skip

import pandas as pd

from rdt.hyper_transformer import HyperTransformer


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

    def test_get_class(self):
        """Get a transformer from its name."""

        # Setup
        ht = HyperTransformer('tests/data/airbnb/airbnb_meta.json')

        # Run
        transformer = ht.get_class('DTTransformer')

        # Check
        assert transformer.__name__ == 'DTTransformer'

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
            columns=['?age', 'age']
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
                assert missing in result.columns
                assert (result[column] == pd.to_numeric(result[column])).all()
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
        ht = HyperTransformer('tests/data/airbnb/airbnb_meta.json')

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
                    assert missing in table.columns
                    assert (table[column] == pd.to_numeric(table[column])).all()

                    if meta_col['type'] != 'categorical':
                        # This is due to the fact that CatTransformer is able to handle
                        # nulls by itself without relying in NullTransformer.
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
            assert table.equals(reversed_table)
