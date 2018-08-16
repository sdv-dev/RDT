import sys

import numpy as np
import pandas as pd

from rdt.transformers.BaseTransformer import BaseTransformer
from rdt.transformers.NullTransformer import NullTransformer


class NumberTransformer(BaseTransformer):
    """
    This class represents the datetime transformer for SDV
    """

    def __init__(self, *args, **kwargs):
        """ initialize transformer """
        super().__init__(type='number', *args, **kwargs)
        self.default_val = None
        self.subtype = None

    def fit_transform(self, col, col_meta, missing=True):
        """ Returns a tuple (transformed_table, new_table_meta) """
        out = pd.DataFrame()
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
        # if are just processing child rows, then the name is already known
        out[self.col_name] = col

        # Handle missing
        if missing:
            nt = NullTransformer()
            out = nt.fit_transform(out, col_meta)
            out[self.col_name] = out.apply(self.get_val, axis=1)
            return out

        out = out.apply(self.get_val, axis=1)
        return out.to_frame(self.col_name)

    def reverse_transform(self, col, col_meta, missing=True):
        """ Converts data back into original format """
        output = pd.DataFrame(columns=[])
        subtype = col_meta['subtype']
        col_name = col_meta['name']
        fn = self.get_number_converter(col_name, subtype)

        if missing:
            new_col = col.apply(fn, axis=1)
            new_col = new_col.rename(col_name)
            data = pd.concat([new_col, col['?' + col_name]], axis=1)
            nt = NullTransformer()
            output[col_name] = nt.reverse_transform(data, col_meta)

        else:
            data = col.to_frame()
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
            val = x[col]
            if np.isposinf(val):
                val = sys.maxsize
            elif np.isneginf(val):
                val = -sys.maxsize
            if meta == 'integer':
                return int(round(val))
            return val

        return safe_round
