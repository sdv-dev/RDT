"""Basic Integration tests for the BaseTransformer."""

import numpy as np
import pandas as pd

from rdt.transformers.base import BaseMultiColumnTransformer, BaseTransformer


def test_dummy_transformer_series_output():
    """Test a transformer that outputs a Series.

    This validates that a Transformer that is implemented to
    produce a Series object as the output of the `_transform`
    method works as expected.

    Setup:
        - A DummyTransformer that implements a _transform
          that produces a Series and a reverse_transform
          that takes a Series as input.

    Input:
        - A DataFrame with a single column of boolean values

    Expected behavior:
        - The data should be transformed into a DataFrame that contains
          a single float column made of 0.0s and 1.0s.
        - The transformed data should be able to reversed to
          re-produce the input data.
    """
    # Setup
    class DummyTransformer(BaseTransformer):

        INPUT_SDTYPE = 'boolean'

        def _fit(self, data):
            pass

        def _transform(self, data):
            return data.astype(float)

        def _reverse_transform(self, data):
            return data.round() != 0

    # Run
    data = pd.DataFrame({
        'bool': [True, False, True, False]
    })

    transformer = DummyTransformer()
    transformed = transformer.fit_transform(data, 'bool')

    reverse = transformer.reverse_transform(transformed)

    # Assert
    expected_transform = pd.DataFrame({
        'bool': [1., 0., 1., 0.]
    })
    pd.testing.assert_frame_equal(expected_transform, transformed)
    pd.testing.assert_frame_equal(reverse, data)


def test_dummy_transformer_dataframe_output():
    """Test a transformer that outputs a DataFrame.

    This validates that a Transformer that is implemented to
    produce a DataFrame object as the output of the `_transform`
    method works as expected.

    Setup:
        - A DummyTransformer that implements a _transform
          that produces a DataFrame and a reverse_transform
          that takes a DataFrame as input.

    Input:
        - A DataFrame with a single column of boolean values

    Expected behavior:
        - The data should be transformed into a DataFrame that contains
          a float column made of 0.0s, 1.0s and -1.0s, and another float
          column that contains 0.0s and 1.0s indicating which values were
          null in the input.
        - The transformed data should be able to reversed to
          re-produce the input data.
    """
    # Setup
    class DummyTransformer(BaseTransformer):

        INPUT_SDTYPE = 'boolean'

        def __init__(self):
            super().__init__()
            self.output_properties = {
                None: {'sdtype': 'float'},
                'null': {'sdtype': 'float'},
            }

        def _fit(self, data):
            pass

        def _transform(self, data):
            out = pd.DataFrame(dict(zip(
                self.output_columns,
                [
                    data.astype(float).fillna(-1),
                    data.isna().astype(float)
                ]
            )))

            return out

        def _reverse_transform(self, data):
            output = data[self.output_columns[0]]
            output = output.round().astype(bool).astype(object)
            output.iloc[data[self.output_columns[1]] == 1] = np.nan

            return output

    data = pd.DataFrame({'bool': [True, False, True, np.nan]})
    transformer = DummyTransformer()

    # Run
    transformed = transformer.fit_transform(data, 'bool')
    reverse = transformer.reverse_transform(transformed)

    # Assert
    expected_transform = pd.DataFrame({
        'bool': [1., 0., 1., -1.],
        'bool.null': [0., 0., 0., 1.]
    })
    pd.testing.assert_frame_equal(expected_transform, transformed)
    pd.testing.assert_frame_equal(reverse, data)


def test_multi_column_transformer_same_number_of_columns_input_output():
    """Test a multi-column transformer when the same of input and output columns."""
    # Setup
    class AdditionTransformer(BaseMultiColumnTransformer):
        """This transformer takes 3 columns and return the cumulative sum of each row."""
        def _fit(self, columns_data, columns_to_sdtypes):
            self.output_properties = {
                column: {'sdtype': 'numerical'} for column in self.columns
            }
            self.dtypes = columns_data.dtypes

        def _transform(self, data):
            return data.cumsum(axis=1)

        def _reverse_transform(self, data):
            result = data.diff(axis=1)
            result.iloc[:, 0] = data.iloc[:, 0]

            return result.astype(self.dtypes)

    data_test = pd.DataFrame({
        'col_1': [1, 2, 3],
        'col_2': [10, 20, 30],
        'col_3': [100, 200, 300]
    })

    column_to_sdtype = {
        'col_1': 'numerical',
        'col_2': 'numerical',
        'col_3': 'numerical'
    }
    transformer = AdditionTransformer()

    # Run
    transformed = transformer.fit_transform(data_test, column_to_sdtype)
    reverse = transformer.reverse_transform(transformed)

    # Assert
    expected_transform = pd.DataFrame({
        'col_1': [1, 2, 3],
        'col_2': [11, 22, 33],
        'col_3': [111, 222, 333]
    })
    pd.testing.assert_frame_equal(expected_transform, transformed)
    pd.testing.assert_frame_equal(reverse, data_test)


def test_multi_column_transformer_less_output_than_input_columns():
    """Test a multi-column transformer when the output has less columns than the input."""
    class ConcatenateTransformer(BaseMultiColumnTransformer):
        """This transformer takes 4 columns and concatenate them into 2 columns.
        The two first and last columns are concatenated together.
        """
        def _fit(self, columns_data, columns_to_sdtypes):
            self.name_1 = self.columns[0] + '#' + self.columns[1]
            self.name_2 = self.columns[2] + '#' + self.columns[3]
            self.output_properties = {
                self.name_1: {'sdtype': 'categorical'},
                self.name_2: {'sdtype': 'categorical'}
            }
            self.dtypes = columns_data.dtypes

        def _transform(self, data):
            data[self.name_1] = data.iloc[:, 0] + '#' + data.iloc[:, 1]
            data[self.name_2] = data.iloc[:, 2] + '#' + data.iloc[:, 3]

            return data.drop(columns=self.columns)

        def _reverse_transform(self, data):
            result = data.copy()
            column_names = list(data.columns)

            col1, col2 = column_names[0].split('#')
            result[[col1, col2]] = result[column_names[0]].str.split('#', expand=True)

            col3, col4 = column_names[1].split('#')
            result[[col3, col4]] = result[column_names[1]].str.split('#', expand=True)

            return result.astype(self.dtypes).drop(columns=column_names)

    data_test = pd.DataFrame({
        'col_1': ['A', 'B', 'C'],
        'col_2': ['D', 'E', 'F'],
        'col_3': ['G', 'H', 'I'],
        'col_4': ['J', 'K', 'L']
    })

    column_to_sdtype = {
        'col_1': 'categorical',
        'col_2': 'categorical',
        'col_3': 'categorical',
        'col_4': 'categorical'
    }
    transformer = ConcatenateTransformer()

    # Run
    transformer.fit(data_test, column_to_sdtype)
    transformed = transformer.transform(data_test)
    reverse = transformer.reverse_transform(transformed)

    # Assert
    expected_transform = pd.DataFrame({
        'col_1#col_2': ['A#D', 'B#E', 'C#F'],
        'col_3#col_4': ['G#J', 'H#K', 'I#L']
    })
    pd.testing.assert_frame_equal(expected_transform, transformed)
    pd.testing.assert_frame_equal(reverse, data_test)


def test_multi_column_transformer_more_output_than_input_columns():
    """Test a multi-column transformer when the output has more columns than the input."""
    class ExpandTransformer(BaseMultiColumnTransformer):

        def _fit(self, columns_data, columns_to_sdtypes):
            name_1 = self.columns[0] + '.first_part'
            name_2 = self.columns[0] + '.second_part'
            name_3 = self.columns[1] + '.first_part'
            name_4 = self.columns[1] + '.second_part'
            self.output_properties = {
                name_1: {'sdtype': 'categorical'},
                name_2: {'sdtype': 'categorical'},
                name_3: {'sdtype': 'categorical'},
                name_4: {'sdtype': 'categorical'}
            }
            self.names = [name_1, name_2, name_3, name_4]
            self.dtypes = columns_data.dtypes

        def _transform(self, data):
            data[self.names[0]] = data[self.columns[0]].str[0]
            data[self.names[1]] = data[self.columns[0]].str[1]
            data[self.names[2]] = data[self.columns[1]].str[0]
            data[self.names[3]] = data[self.columns[1]].str[1]

            return data.drop(columns=self.columns)

        def _reverse_transform(self, data):
            result = data.copy()
            result[self.columns[0]] = result[self.names[0]] + result[self.names[1]]
            result[self.columns[1]] = result[self.names[2]] + result[self.names[3]]

            return result.astype(self.dtypes).drop(columns=self.names)

    data_test = pd.DataFrame({
        'col_1': ['AB', 'CD', 'EF'],
        'col_2': ['GH', 'IJ', 'KL'],
    })

    column_to_sdtype = {
        'col_1': 'categorical',
        'col_2': 'categorical',
    }
    transformer = ExpandTransformer()

    # Run
    transformer.fit(data_test, column_to_sdtype)
    transformed = transformer.transform(data_test)
    reverse = transformer.reverse_transform(transformed)

    # Assert
    expected_transform = pd.DataFrame({
        'col_1.first_part': ['A', 'C', 'E'],
        'col_1.second_part': ['B', 'D', 'F'],
        'col_2.first_part': ['G', 'I', 'K'],
        'col_2.second_part': ['H', 'J', 'L']
    })
    pd.testing.assert_frame_equal(expected_transform, transformed)
    pd.testing.assert_frame_equal(reverse, data_test)
