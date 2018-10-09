import numpy as np
import pandas as pd

from rdt.transformers.BaseTransformer import BaseTransformer


class NullTransformer(BaseTransformer):
    """
    This class represents the datetime transformer for SDV
    """

    def __init__(self, *args, **kwargs):
        """ initialize transformer """
        super().__init__(type=['datetime', 'number'], *args, **kwargs)

    def fit_transform(self, col, col_meta, **kwargs):
        """ Returns a tuple (transformed_table, new_table_meta) """
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
        """ Converts data back into original format """
        output = pd.DataFrame(columns=[])
        col_name = col_meta['name']
        fn = self.get_null_converter(col_name)
        output[col_name] = col.apply(fn, axis=1)
        return output

    def get_null_converter(self, col):
        '''Returns a converter that takes a value and replaces
        it with null if it's supposed to be missing
        :param col: name of column
        :type col: str
        :param meta: type of column values
        :type meta: str

        :returns: function
        '''

        def nullify(x):
            val = x[col]
            try:
                if x['?' + col] == 0:
                    return np.nan
            except Exception as inst:
                print(inst)
            return val

        return nullify
