import numpy as np
import pandas as pd

from rdt.transformers.base import BaseTransformer


class NullTransformer(BaseTransformer):
    """Transformer for missing/null data."""

    type = ['datetime', 'number', 'categorical']

    def __init__(self, column_metadata):
        """ initialize transformer """
        super().__init__(column_metadata)

    def fit(self, col):
        self.new_name = '?' + self.col_name

        if isinstance(col, pd.DataFrame):
            col = col[self.col_name]

        if self.column_metadata['type'] == 'number':
            mean = col.mean()

            if pd.notnull(mean):
                self.default_value = mean

            else:
                self.default_value = 0
        else:
            self.default_value = col.mode().iloc[0]

    def transform(self, col):
        """Prepare the transformer to convert data and return the processed table.

        Args:
            col (pandas.DataFrame):
                Data to transform.

        Returns:
            pandas.DataFrame
        """
        out = pd.DataFrame(index=col.index)
        out[self.col_name] = col.fillna(self.default_value)
        out[self.new_name] = (pd.notnull(col) * 1).astype(int)
        return out

    def reverse_transform(self, col):
        """Converts data back into original format.

        Args:
            col (pandas.DataFrame):
                Data to transform.

        Returns:
            pandas.DataFrame
        """
        output = pd.DataFrame()
        new_name = '?' + self.col_name

        col.loc[col[new_name] == 0, self.col_name] = np.nan
        output[self.col_name] = col[self.col_name]
        return output
