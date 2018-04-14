import numpy as np
import pandas as pd

from dataprep.transformers.BaseTransformer import *


class NumberTransformer(BaseTransformer):
    """
    This class represents the datetime transformer for SDV
    """

    def __init__(self):
        """ initialize transformer """
        super(NumberTransformer, self).__init__()
        self.type = 'number'
        self.missing = None
        self.col_name = None
        self.default_val = None
        self.subtype = None

    def fit_transform(self, col, col_meta):
        """ Returns a tuple (transformed_table, new_table_meta) """
        out = pd.DataFrame(columns=[])
        self.col_name = col_meta['name']
        self.subtype = col_meta['subtype']
        # get default val
        for x in col:
            if x is not None:
                try:
                    tmp = int(round(x))
                    self.default_val = tmp
                    break
                except Exception as inst:
                    continue
        # create an extra column for missing values if they exist in the data
        new_name = '?' + self.col_name
        # if are just processing child rows, then the name is already known
        self.missing = pd.notnull(col) * 1
        out[new_name] = self.missing
        out[self.col_name] = col
        out = out.apply(self.get_val, axis=1)
        return out.to_frame(self.col_name)

    def transform(self, col, col_meta):
        """ Does the required transformations to the data """
        return self.fit_transform(col, col_meta)

    def reverse_transform(self, col, col_meta):
        """ Converts data back into original format """
        output = pd.DataFrame(columns=[])
        subtype = col_meta['subtype']
        col_name = col_meta['name']
        fn = self.get_number_converter(col_name, subtype)
        data = col.to_frame()
        data['?'+col_name] = self.missing
        output[col_name] = data.apply(fn, axis=1)
        return output

    def get_val(self, x):
        """ Converts to int """
        try:
            if self.subtype == 'integer':
                return int(round(x[self.col_name]))
            else:
                if np.isnan(x[self.col_name]):
                    return self.default_val
                return x[self.col_name]
        except (ValueError, TypeError):
            return self.default_val

    def get_number_converter(self, col, meta):
        '''Returns a converter that takes in a value and turns it into an
           integer, if necessary

        :param col: name of column
        :type col: str
        :param missing: true if column has NULL values
        :type missing: bool
        :param meta: type of column values
        :type meta: str

        :returns: function
        '''

        def safe_round(x):
            if meta == 'integer':
                if x['?' + col] == 0:
                    return np.nan
                return int(round(x[col]))
            return x[col]

        return safe_round
