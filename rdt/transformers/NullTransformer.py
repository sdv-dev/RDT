import numpy as np
import pandas as pd

from rdt.transformers.BaseTransformer import BaseTransformer


class NullTransformer(BaseTransformer):
    """Transformer for missing/null data."""

    def __init__(self, *args, **kwargs):
        """ initialize transformer """
        super().__init__(type=['datetime', 'number'], *args, **kwargs)

    def fit_transform(self, col, col_meta, **kwargs):
        """Prepare the transformer to convert data and return the processed table.

        Args:
            col(pandas.DataFrame): Data to transform.
            col_meta(dict): Meta information of the column.
            missing(bool): Wheter or not handle missing values using NullTransformer.

        Returns:
            pandas.DataFrame
        """
        out = pd.DataFrame(columns=[])
        self.col_name = col_meta['name']

        # create an extra column for missing values if they exist in the data
        new_name = '?' + self.col_name
        out[new_name] = (pd.notnull(col) * 1).astype(int)

        if isinstance(col, pd.DataFrame):
            null_mean = pd.isnull(col.mean()).all()
        else:
            null_mean = pd.isnull(col.mean())
        # replace missing values
        if col_meta['type'] == 'number' and not null_mean:
            clean_col = col.fillna(col.mean())
        else:
            clean_col = col.fillna(0)

        out[self.col_name] = clean_col
        return out

    def reverse_transform(self, col, col_meta):
        """Converts data back into original format.

        Args:
            col(pandas.DataFrame): Data to transform.
            col_meta(dict): Meta information of the column.
            missing(bool): Wheter or not handle missing values using NullTransformer.

        Returns:
            pandas.DataFrame
        """
        output = pd.DataFrame(columns=[])
        col_name = col_meta['name']
        fn = self.get_null_converter(col_name)
        output[col_name] = col.apply(fn, axis=1)
        return output

    def get_null_converter(self, col_name):
        """Return a function that take a row replaces it with null if it's supposed to be missing.

        Args:
            col_name(str): Name of the column.

        Returns:
            function
        """

        def nullify(x):
            val = x[col_name]
            try:
                if x['?' + col_name] == 0:
                    return np.nan
            except Exception as inst:
                print(inst)
            return val

        return nullify
