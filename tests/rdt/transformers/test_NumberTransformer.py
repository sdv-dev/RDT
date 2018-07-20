import unittest

import pandas as pd
import numpy as np

from rdt.transformers.NumberTransformer import NumberTransformer


class Test_NumberTransformer(unittest.TestCase):
    def test___init__(self):
        """On init, sets type to number"""

        # Run
        transformer = NumberTransformer()

        # Check
        assert (transformer.type == 'number')

    def test_fit_transform(self):
        """FIXME: finish when number and null transformer are compatible"""
        """ fit_transform sets internal state and transforms data """

        # Setup
        col = pd.Series([62, 27, 5, 34, 62])
        col_meta = {
            "name": "age",
            "type": "number",
            "subtype": "integer",
        }
        transformer = NumberTransformer()
        data = {'age': [62, 27, 5, 34, 62]}

        # Run
        result = pd.DataFrame(data)
        expected_result = transformer.fit_transform(col, col_meta, False)

        # Check
        assert result.equals(expected_result)

    @unittest.skip("FIXME: when number and null transformer are compatible")
    def test_fit_transform_missing(self):
        """FIXME: finish when number and null transformer are compatible"""
        """ Sets internal state and transforms data with missing values"""

        # Setup
        col = pd.Series([62, 27, np.nan, 34, 62])
        col_meta = {
            "name": "age",
            "type": "number",
            "subtype": "integer",
        }
        transformer = NumberTransformer()
        data = {'age': [62, 27, 0, 34, 62]}

        # Run
        result = pd.DataFrame(data)
        expected_result = transformer.fit_transform(col, col_meta, True)

        # Check
        assert result.age.equals(expected_result.age)

    def test_reverse_transform(self):
        """FIXME: finish when number and null transformer are compatible"""
        """ Checks the conversion of the data back into original format """

        # Setup
        col = pd.Series([34, 23, 27, 31, 39], name='age')
        col_meta = {
            'name': 'age',
            'subtype': 'integer',
            'type': 'number'
        }
        transformer = NumberTransformer()

        # Run
        result = pd.DataFrame({'age': [34, 23, 27, 31, 39]})
        expected_result = transformer.reverse_transform(col, col_meta, False)

        # Check
        assert result.equals(expected_result)

    @unittest.skip("FIXME: when number and null transformer are compatible")
    def test_reverse_transform_missing(self):
        """FIXME: finish when number and null transformer are compatible"""
        """ Sets internal state and transforms data with missing values"""

        # Setup
        col = pd.DataFrame({
            'age': [34, 23, 0, 31, 39],
            '?age': [1, 1, 0, 1, 1]
        })
        col_meta = {
            'name': 'age',
            'subtype': 'integer',
            'type': 'number'
        }
        transformer = NumberTransformer()

        # Run
        result = pd.DataFrame({'age': [34, 23, np.nan, 31, 39]})
        expected_result = transformer.reverse_transform(col, col_meta, True)

        # Check
        assert result.equals(expected_result)

    def test_get_val_subtype_integer(self):
        """ Rounds off the value and returns it """

        # Setup
        col = pd.DataFrame({
            'age': [62, 35, 24]
        })

        transformer = NumberTransformer()
        transformer.col_name = 'age'
        transformer.subtype = 'integer'
        transformer.default_val = col.age[0]
        expected_result = pd.Series([62, 35, 24])

        # Run
        result = col.apply(transformer.get_val, axis=1)

        # Check
        assert result.equals(expected_result)

    def test_get_val_subtype_not_integer(self):
        """ If subtype is not integer and there are not null values,
        returns the same value """

        # Setup
        col = pd.DataFrame({
            'age': [62.5, 35.5, 24.3]
        })

        transformer = NumberTransformer()
        transformer.col_name = 'age'
        transformer.subtype = 'decimal'
        transformer.default_val = col.age[0]
        expected_result = pd.Series([62.5, 35.5, 24.3])

        # Run
        result = col.apply(transformer.get_val, axis=1)

        # Check
        assert result.equals(expected_result)

    def test_get_val_null_value(self):
        """ If subtype is not integer and there are null values,
         returns the default value """

        # Setup
        col = pd.DataFrame({
            'age': [4, np.nan, np.nan]
        })

        transformer = NumberTransformer()
        transformer.col_name = 'age'
        transformer.subtype = 'decimal'
        transformer.default_val = col.age[0]
        expected_result = pd.Series([4.0, 4.0, 4.0])
        # Run
        result = col.apply(transformer.get_val, axis=1)

        # Check
        assert result.equals(expected_result)

    def test_get_val_error(self):
        """ Returns 'default_val' when there is a ValueError """
        # Setup
        col = pd.DataFrame({
            'age': [4, 'hoo', 13]
        })

        transformer = NumberTransformer()
        transformer.col_name = 'age'
        transformer.subtype = 'integer'
        transformer.default_val = 999
        expected_result = pd.Series([4, 999, 13])

        # Run
        result = col.apply(transformer.get_val, axis=1)

        # Check
        assert result.equals(expected_result)

    def test_get_number_converter(self):
        """ If meta 'integer', cast values to intif not, returns as is """
        # Setup
        transformer = NumberTransformer()
        col_name = 'age'
        converter = transformer.get_number_converter(col_name, 'integer')
        data = pd.DataFrame({
            'age': [0.5, 10.1, 3]
        })
        expected_result = pd.Series([0, 10, 3], name='age')

        # Run
        result = data.apply(converter, axis=1)

        # Check
        assert result.equals(expected_result)
