from unittest import TestCase, mock, skip

import numpy as np
import pandas as pd

from rdt.transformers.CatTransformer import CatTransformer


class TestCatTransformer(TestCase):

    def test___init__(self):
        """After parent init set type and probability_map."""

        # Run
        transformer = CatTransformer()

        # Check
        assert transformer.type == 'categorical'
        assert transformer.probability_map == {}

    @skip("https://github.com/HDI-Project/RDT/issues/25")
    def test_fit_transform(self):
        """fit_transform sets internal state and transforms data."""

        # Setup
        transformer = CatTransformer()
        col = pd.Series(['B', 'B', 'A', 'B', 'A'])
        col_meta = {
            "name": "breakfast",
            "type": "categorical"
        }

        expected_result = pd.DataFrame({
            'breakfast': [0.7, 0.7, 0.2, 0.7, 0.2]
        })

        # Run
        result = transformer.fit_transform(col, col_meta, False)

        # Check
        assert result.equals(expected_result)

    @skip("https://github.com/HDI-Project/RDT/issues/25")
    def test_fit_transform_missing(self):
        """fit_transform sets internal state and transforms data with null values."""

        # Setup
        transformer = CatTransformer()
        original_column = pd.Series(['B', 'B', 'A', 'B', 'A'])
        col_meta = {
            "name": "breakfast",
            "type": "categorical"
        }

        # Run
        result = transformer.fit_transform(original_column, col_meta, missing=True)

        # Check
        assert original_column.equals(result)

    def test_reverse_transform(self):
        """reverse_transform change back the data into original format."""

        # Setup
        transformer = CatTransformer()
        transformer.probability_map = {
            'A': ((0.6, 1.0), 0.8, 0.0666),
            'B': ((0, 0.6), 0.3, 0.0999)
        }
        col_meta = {
            "name": "breakfast",
            "type": "categorical"
        }
        col = pd.DataFrame({
            'breakfast': [0.1, 0.4, 0.8, 0.3, 0.7]
        })
        expected_result = pd.DataFrame({
            'breakfast': ['B', 'B', 'A', 'B', 'A']
        })

        # Run
        result = transformer.reverse_transform(col, col_meta, False)

        # Check
        assert result.equals(expected_result)

    def test_reverse_transform_missing(self):
        """Changes back the data into original format."""

        # Setup
        transformer = CatTransformer()
        transformer.probability_map = {
            'A': ((0.6, 1.0), 0.8, 0.0666),
            'B': ((0, 0.6), 0.3, 0.0999)
        }
        col_meta = {
            "name": "breakfast",
            "type": "categorical"
        }
        col = pd.DataFrame({
            'breakfast': [0.2, 0.4, 0.8, 0.3, 0.7],
            '?breakfast': [1, 1, 1, 1, 1]
        })
        expected_result = pd.DataFrame({
            'breakfast': ['B', 'B', 'A', 'B', 'A']
        })

        # Run
        result = transformer.reverse_transform(col, col_meta, True)

        # Check
        assert result.equals(expected_result)

    @mock.patch('scipy.stats.norm.rvs')
    def test_get_val(self, rvs_mock):
        """Checks the random value."""

        # Setup
        transformer = CatTransformer()
        transformer.probability_map = {
            'A': ((0.6, 1.0), 0.8, 0.0666),
            'B': ((0, 0.6), 0.3, 0.0999)
        }
        rvs_mock.return_value = 1

        # Run
        result = transformer.get_val('B')

        # Check
        assert result == 1

    def test_get_reverse_cat(self):
        """get_reverse_cat return a function that returns the category from a numerical value."""

        # Setup
        original_column = pd.DataFrame({
            'breakfast': ['B', 'B', 'A', 'B', 'A']
        })
        col_meta = {
            "name": "breakfast",
            "type": "categorical"
        }
        transformer = CatTransformer()
        transformed_data = transformer.fit_transform(original_column, col_meta, False)

        # Run
        converter = transformer.get_reverse_cat('breakfast')
        result = transformed_data.apply(converter, axis=1)

        # Check
        assert (result == original_column['breakfast']).all()

    def test_get_probability_map(self):
        """Maps the values to probabilities."""

        # Setup
        data = pd.DataFrame({
            'col1': ['A', 'B', 'A', 'B', 'B']
        })
        transformer = CatTransformer()
        transformer.col_name = 'col1'

        # Run
        transformer.get_probability_map(data)

        # Check
        # Keys are unique values of initial data
        assert set(transformer.probability_map.keys()) == set(data['col1'].unique())

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
        col_meta = {
            "name": "breakfast",
            "type": "categorical"
        }
        transformer = CatTransformer()

        # Run
        transformer.fit_transform(data, col_meta)

        # Check
        # The nan value in the data should be in probability map
        assert None in transformer.probability_map
