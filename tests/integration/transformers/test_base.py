"""Basic Integration tests for the BaseTransformer."""

import numpy as np
import pandas as pd

from rdt.transformers.base import BaseTransformer


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

        INPUT_TYPE = 'boolean'
        OUTPUT_TYPES = {
            'value': 'float'
        }

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
        'bool.value': [1., 0., 1., 0.]
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

        INPUT_TYPE = 'boolean'
        OUTPUT_TYPES = {
            'value': 'float',
            'null': 'float'
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

    # Run
    data = pd.DataFrame({
        'bool': [True, False, True, np.nan]
    })

    transformer = DummyTransformer()
    transformed = transformer.fit_transform(data, 'bool')

    reverse = transformer.reverse_transform(transformed)

    # Assert
    expected_transform = pd.DataFrame({
        'bool.value': [1., 0., 1., -1.],
        'bool.null': [0., 0., 0., 1.]
    })
    pd.testing.assert_frame_equal(expected_transform, transformed)
    pd.testing.assert_frame_equal(reverse, data)


def test_dummy_transformer_multi_column_input():
    """Test a transformer that inputs a DataFrame.

    This validates that a Transformer that is implemented to
    expect multiple columns as the input for the `_transform`
    method works as expected.

    Setup:
        - A DummyTransformer that implements a _transform
          that expects a DataFrame with multiple columns as
          input, and a reverse transform that produces the
          same set of columns.

    Input:
        - A DataFrame with three year, month and day columns that
          represent dates.

    Expected behavior:
        - The data should be transformed into a DataFrame that contains
          a single float column with the timestamp values of the inputed
          datetimes.
        - The data should be able to be transformed and reverse
          transformed to re-produce the input data.
    """
    # Setup
    class DummyTransformer(BaseTransformer):

        INPUT_TYPE = 'datetime'
        OUTPUT_TYPES = {
            'value': 'float',
        }

        def _fit(self, data):
            pass

        def _transform(self, data):
            # Convert multiple columns into a single datetime
            data = pd.to_datetime(data)

            float_data = data.to_numpy().astype(float)
            data_is_nan = data.isna().to_numpy().astype(float)

            output = dict(zip(
                self.output_columns,
                [float_data, data_is_nan]
            ))

            output = pd.DataFrame(output).fillna(-1)

            return output

        def _reverse_transform(self, data):
            datetimes = data.round().astype('datetime64[ns]')
            out = pd.DataFrame({
                'year': datetimes.dt.year,
                'month': datetimes.dt.month,
                'day': datetimes.dt.day,
            })

            return out

    # Run
    data = pd.DataFrame({
        'year': [2001, 2002, 2003],
        'month': [1, 2, 3],
        'day': [1, 2, 3],
    })

    transformer = DummyTransformer()
    transformed = transformer.fit_transform(data, ['year', 'month', 'day'])

    reverse = transformer.reverse_transform(transformed)

    # Assert
    expected_transform = pd.DataFrame({
        'year#month#day.value': [9.783072e+17, 1.012608e+18, 1.046650e+18]
    })
    pd.testing.assert_frame_equal(expected_transform, transformed)
    pd.testing.assert_frame_equal(reverse, data)
