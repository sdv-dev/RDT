from unittest import TestCase, skip
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from faker import Faker

from rdt.transformers.category import CatTransformer


class TestCatTransformer(TestCase):

    def test___init__(self):
        """On init, anonymize and category args are set as attributes."""
        # Setup
        column_metadata = {
            "name": "breakfast",
            "type": "categorical"
        }

        # Run
        transformer = CatTransformer(column_metadata)

        # Check
        assert transformer.type == 'categorical'
        assert transformer.probability_map == {}
        assert transformer.anonymize is None
        assert transformer.category is None

    def test___init___anonymize_without_category_raises(self):
        """On init, if anonymize is True, category is required."""
        # Setup
        column_metadata = {
            "name": "breakfast",
            "type": "categorical",
            "pii": True
        }

        # Run / Check
        with self.assertRaises(ValueError):
            CatTransformer(column_metadata)

    def test___init___category_not_suported_raises(self):
        """On init, if anonymize is True and category is not supported, and exception raises."""
        # Setup
        column_metadata = {
            "name": "breakfast",
            "type": "categorical",
            "pii": True,
            "pii_category": "blabla"
        }

        # Run / Check
        with self.assertRaises(ValueError):
            CatTransformer(column_metadata)

    @patch('rdt.transformers.category.Faker')
    def test_get_generator(self, faker_mock):
        """get_generator return a function to create new values for a category."""
        # Setup
        column_metadata = {
            "name": "breakfast",
            "type": "categorical",
            "pii": True,
            "pii_category": "first_name"
        }
        transformer = CatTransformer(column_metadata)
        faker_instance = MagicMock(spec=Faker())
        faker_mock.return_value = faker_instance

        expected_call_args_list = [(), ()]

        # Run
        result = transformer.get_generator()

        # Check
        assert faker_mock.call_args_list == expected_call_args_list
        assert result == faker_instance.first_name

    @patch('rdt.transformers.category.Faker')
    def test_get_generator_raises_unsupported(self, faker_mock):
        """If the category is not supported, raise an exception."""
        # Setup
        column_metadata = {
            "name": "breakfast",
            "type": "categorical",
            "pii": True,
            "pii_category": "superhero_names"
        }
        transformer = CatTransformer(column_metadata)
        faker_instance = MagicMock(spec=Faker())
        faker_mock.return_value = faker_instance

        # Run / Check
        with self.assertRaises(ValueError):
            transformer.get_generator()

    @skip("https://github.com/HDI-Project/RDT/issues/25")
    def test_fit_transform(self):
        """fit_transform sets internal state and transforms data."""

        # Setup
        column_metadata = {
            "name": "breakfast",
            "type": "categorical"
        }
        transformer = CatTransformer(column_metadata)
        col = pd.Series(['B', 'B', 'A', 'B', 'A'])
        column_metadata = {
            "name": "breakfast",
            "type": "categorical"
        }

        expected_result = pd.DataFrame({
            'breakfast': [0.7, 0.7, 0.2, 0.7, 0.2]
        })

        # Run
        result = transformer.fit_transform(col)

        # Check
        assert result.equals(expected_result)

    @skip("https://github.com/HDI-Project/RDT/issues/25")
    def test_fit_transform_missing(self):
        """fit_transform sets internal state and transforms data with null values."""

        # Setup
        column_metadata = {
            "name": "breakfast",
            "type": "categorical"
        }
        transformer = CatTransformer(column_metadata)
        original_column = pd.Series(['B', 'B', 'A', 'B', 'A'])
        column_metadata = {
            "name": "breakfast",
            "type": "categorical"
        }

        # Run
        result = transformer.fit_transform(original_column)

        # Check
        assert original_column.equals(result)

    def test_reverse_transform(self):
        """reverse_transform change back the data into original format."""

        # Setup
        column_metadata = {
            "name": "breakfast",
            "type": "categorical"
        }
        transformer = CatTransformer(column_metadata)
        transformer.probability_map = {
            'A': ((0.6, 1.0), 0.8, 0.0666),
            'B': ((0, 0.6), 0.3, 0.0999)
        }
        transformer.col_name = 'breakfast'

        col = pd.DataFrame({
            'breakfast': [0.1, 0.4, 0.8, 0.3, 0.7]
        })
        expected_result = pd.DataFrame({
            'breakfast': ['B', 'B', 'A', 'B', 'A']
        })

        # Run
        result = transformer.reverse_transform(col)

        # Check
        assert result.equals(expected_result)

    def test_reverse_transform_missing(self):
        """Changes back the data into original format."""

        # Setup
        column_metadata = {
            "name": "breakfast",
            "type": "categorical"
        }
        transformer = CatTransformer(column_metadata)
        transformer.probability_map = {
            'A': ((0.6, 1.0), 0.8, 0.0666),
            'B': ((0, 0.6), 0.3, 0.0999)
        }
        transformer.col_name = 'breakfast'

        col = pd.DataFrame({
            'breakfast': [0.2, 0.4, 0.8, 0.3, 0.7],
            '?breakfast': [1, 1, 1, 1, 1]
        })
        expected_result = pd.DataFrame({
            'breakfast': ['B', 'B', 'A', 'B', 'A']
        })

        # Run
        result = transformer.reverse_transform(col)

        # Check
        assert result.equals(expected_result)

    @patch('scipy.stats.norm.rvs')
    def test_get_val(self, rvs_mock):
        """Checks the random value."""

        # Setup
        column_metadata = {
            "name": "breakfast",
            "type": "categorical"
        }
        transformer = CatTransformer(column_metadata)
        transformer.probability_map = {
            'A': ((0.6, 1.0), 0.8, 0.0666),
            'B': ((0, 0.6), 0.3, 0.0999)
        }
        rvs_mock.return_value = 1

        # Run
        result = transformer.get_val('B')

        # Check
        assert result == 1

    def test_get_category(self):
        """get_category return the category from a numerical value."""

        # Setup
        original_column = pd.DataFrame({
            'breakfast': ['B', 'B', 'A', 'B', 'A']
        })
        column_metadata = {
            "name": "breakfast",
            "type": "categorical"
        }
        transformer = CatTransformer(column_metadata, False)
        transformed_data = transformer.fit_transform(original_column)

        # Run
        result = transformer.get_category(transformed_data['breakfast'])

        # Check
        assert (result == original_column['breakfast']).all()

    def test_fit(self):
        """Maps the values to probabilities."""
        # Setup
        column_metadata = {
            "name": "breakfast",
            "type": "categorical"
        }
        data = pd.DataFrame({
            'breakfast': ['A', 'B', 'A', 'B', 'B']
        })
        transformer = CatTransformer(column_metadata)

        # Run
        transformer.fit(data)

        # Check
        # Keys are unique values of initial data
        assert set(transformer.probability_map.keys()) == set(data['breakfast'].unique())

        frequency = {  # The frequency of the values in data
            'A': 0.4,
            'B': 0.6
        }

        for key in transformer.probability_map.keys():
            with self.subTest(key=key):
                values = transformer.probability_map[key]
                interval = values[0]
                mean = values[1]

                # Length of interval is frequency
                assert interval[1] - interval[0] == frequency[key]

                # Mean is middle point
                # We check this way because of floating point issues
                assert (mean - interval[0]) - (interval[1] - mean) < 1 / 1E9

    def test_fit_transform_val_nan(self):
        """Tests that nans are handled by fit_transform method."""

        # Setup
        data = pd.DataFrame({
            'breakfast': [np.nan, 1, 5]
        })
        column_metadata = {
            "name": "breakfast",
            "type": "categorical"
        }
        transformer = CatTransformer(column_metadata)

        # Run
        transformer.fit_transform(data)

        # Check
        # The nan value in the data should be in probability map
        assert None in transformer.probability_map

    @patch('rdt.transformers.category.Faker')
    def test_fit_transform_anonymize(self, faker_mock):
        """If anonymize is True the values are replaced before generating probability_map."""
        # Setup
        column_metadata = {
            'name': 'first_name',
            'type': 'categorical',
            'pii': True,
            'pii_category': 'first_name'
        }
        transformer = CatTransformer(column_metadata)

        data = pd.DataFrame({'first_name': ['Albert', 'John', 'Michael']})

        faker_instance = MagicMock()
        faker_instance.first_name.side_effect = ['Anthony', 'Charles', 'Mark']
        faker_mock.return_value = faker_instance

        # Run
        result = transformer.fit_transform(data)

        # Check
        assert set(transformer.probability_map.keys()) == set(['Anthony', 'Charles', 'Mark'])
        assert result.shape == data.shape

    def test_anonymize_not_reversible(self):
        """If anonymize is True the operation is not reversible. """
        # Setup
        regular_column_metadata = {
            'name': 'first_name',
            'type': 'categorical'
        }
        transformer = CatTransformer(regular_column_metadata)

        pii_column_metadata = regular_column_metadata.copy()
        pii_column_metadata.update({
            'pii': True,
            'pii_category': 'first_name'
        })
        anon_transformer = CatTransformer(pii_column_metadata)

        data = pd.DataFrame({
            'first_name': ['Albert', 'John', 'Michael']
        })

        # Run
        transformed = transformer.fit_transform(data)
        reverse_transformed = transformer.reverse_transform(transformed)

        transformed_anonymized = anon_transformer.fit_transform(data)
        reverse_transformed_anonymized = anon_transformer.reverse_transform(transformed_anonymized)

        # Check
        assert data.equals(reverse_transformed)
        assert not data.equals(reverse_transformed_anonymized)
