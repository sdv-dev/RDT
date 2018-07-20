import unittest
from unittest import mock

import pandas as pd

from rdt.transformers.CatTransformer import CatTransformer


class Test_CatTransformer(unittest.TestCase):

    def test___init__(self):
        """ After parent init sets type and probability_map """

        # Run
        transformer = CatTransformer()

        # Check
        assert transformer.type == 'categorical'
        assert transformer.probability_map == {}

    @unittest.skip("FIXME: Accessing unnordered dictionary")
    def test_fit_transform(self):
        """ fit_transform sets internal state and transforms data """

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

    @unittest.skip("FIXME: Categorical and null transformer are compatible")
    def test_fit_transform_missing(self):
        """ fit_transform sets internal state and transforms data with null values """
        # FIXME: finish when number and null transformer are compatible

        # Setup
        transformer = CatTransformer()
        col = pd.Series(['B', 'B', 'A', 'B', 'A'])
        col_meta = {
            "name": "breakfast",
            "type": "categorical"
        }

        result = pd.DataFrame()

        # Run
        expected_result = transformer.fit_transform(col, col_meta, missing=True)

        # Check
        assert result.equals(expected_result)

    def test_reverse_transform(self):
        """ Changes back the data into original format """

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
        col = pd.Series([0.1, 0.4, 0.8, 0.3, 0.7])
        result = pd.DataFrame({
            'breakfast': ['B', 'B', 'A', 'B', 'A']
        })

        # Run
        expected_result = transformer.reverse_transform(col, col_meta, False)

        # Check
        assert result.equals(expected_result)

    @unittest.skip("FIXME: Categorical and null transformer are compatible")
    def test_reverse_transform_missing(self):
        """ Changes back the data into original format """
        """FIXME: finish when number and null transformer are compatible"""

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
        col = pd.Series([0.2, 0.4, 0.8, 0.3, 0.7])
        result = pd.DataFrame({
            'breakfast': ['B', 'B', 'A', 'B', 'A']
        })

        # Run
        expected_result = transformer.reverse_transform(col, col_meta, True)

        # Check
        assert result.equals(expected_result)

    @mock.patch('scipy.stats.norm.rvs')
    def test_get_val(self, rvs_mock):
        """ Checks the random value """

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
        """ Gets a value and returns it back to category """

        # Setup
        col = pd.Series(['B', 'B', 'A', 'B', 'A'])
        col_meta = {
            "name": "breakfast",
            "type": "categorical"
        }
        transformer = CatTransformer()

        # Run
        data = transformer.fit_transform(col, col_meta, False)
        converter = transformer.get_reverse_cat(col)
        expected_result = data.apply(converter, axis=1)

        # Check
        assert col.equals(expected_result)

    def test_get_probability_map(self):
        """ Maps the values to probabilities """

        # Setup
        data = pd.Series(['A', 'B', 'A', 'B', 'B'])
        transformer = CatTransformer()

        # Run
        transformer.get_probability_map(data)

        # Check
        # Keys are unique values of initial data
        assert list(transformer.probability_map.keys()) == list(data.unique())

        frequency = {  # The frequency of the values in data
            'A': 0.4,
            'B': 0.6
        }

        for key in transformer.probability_map.keys():
            values = transformer.probability_map[key]
            interval = values[0]
            mean = values[1]

            # Length of interval is frequency
            assert interval[1] - interval[0] == frequency[key]

            # Mean is middle point
            # We check this way because of floating point issues
            assert (mean - interval[0]) - (interval[1] - mean) < 1 / 1E9
