import numpy as np
import pandas as pd

from sklearn.preprocessing import Imputer
from rdt.transformers.BaseTransformer import BaseTransformer


class NullTransformer(BaseTransformer):
    """
    This class represents the datetime transformer for SDV
    """

    def __init__(self):
        """ initialize transformer """
        super(NullTransformer, self).__init__()
        self.type = ['datetime', 'number']
        self.col_name = None

    def fit_transform(self, col, col_meta):
        """ Returns a tuple (transformed_table, new_table_meta) """
        out = pd.DataFrame(columns=[])
        self.col_name = col_meta['name']
        # replace missing values
        # create an extra column for missing values if they exist in the data
        new_name = '?' + self.col_name
        # if are just processing child rows, then the name is already known
        out[new_name] = pd.notnull(col) * 1
        imp = Imputer()
        data = col.as_matrix().reshape(-1, 1)
        out[self.col_name] = imp.fit_transform(data)
        return out

    def transform(self, col, col_meta):
        """ Does the required transformations to the data """
        return self.fit_transform(col, col_meta)

    def reverse_transform(self, col, col_meta):
        """ Converts data back into original format """
        output = pd.DataFrame(columns=[])
        col_name = col_meta['name']
        fn = self.get_null_converter(col_name)
        print('column', col)
        print('type', type(col))
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
