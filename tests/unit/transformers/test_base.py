import pandas as pd

from rdt.transformers.base import BaseTransformer


class TestBaseTransformer:

    def test__set_columns_data_preserves_order_series(self):
        """Test the ``_set_columns_data`` method.

        The method should not reorder the rows from the ``columns_data``
        parameter if it is a ``Series`` and the ``data`` has a different index.

        Input:
            - data will be a DataFrame with a non-sequential index.
            - columns_data will be a Series with a sequential index.
            - columns will have the column name of the Series.

        Expected behavior:
            - Data should have the values from columns_data in the same order
            as they were in columns_data.
        """
        data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [1, 2, 3]
        }, index=[2, 0, 1])
        columns = ['c']
        columns_data = pd.Series([7, 8, 9], name='c')

        BaseTransformer._set_columns_data(data, columns_data, columns)

        expected = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        }, index=[2, 0, 1])
        pd.testing.assert_frame_equal(data, expected)

    def test__set_columns_data_preserves_order_dataframe(self):
        """Test the ``_set_columns_data`` method.

        The method should not reorder the rows from the ``columns_data``
        parameter if it is a ``DataFrame`` and the ``data`` has a different index.

        Input:
            - data will be a DataFrame with a non-sequential index.
            - columns_data will be a Series with a sequential index.
            - columns will have the column name of the Series.

        Expected behavior:
            - Data should have the values from columns_data in the same order
            as they were in columns_data.
        """
        data = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [1, 2, 3]
        }, index=[2, 0, 1])
        columns = ['c', 'd']
        columns_data = pd.DataFrame({
            'c': [7, 8, 9],
            'd': [10, 11, 12]
        })

        BaseTransformer._set_columns_data(data, columns_data, columns)

        expected = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9],
            'd': [10, 11, 12]
        }, index=[2, 0, 1])
        pd.testing.assert_frame_equal(data, expected)
