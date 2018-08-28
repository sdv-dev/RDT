import sys
import time

import numpy as np
import pandas as pd
from dateutil import parser

from rdt.transformers.BaseTransformer import BaseTransformer
from rdt.transformers.NullTransformer import NullTransformer


class DTTransformer(BaseTransformer):
    """
    This class represents the datetime transformer for SDV
    """

    def __init__(self, *args, **kwargs):
        """ initialize transformer """
        super().__init__(type='datetime', *args, **kwargs)
        self.default_val = None

    def fit_transform(self, col, col_meta, missing=True):
        """ Returns a tuple (transformed_table, new_table_meta) """
        out = pd.DataFrame()
        self.col_name = col_meta['name']

        # get default val
        val = col.groupby(col).count().index[0]
        tmp = parser.parse(str(val)).timetuple()
        self.default_val = time.mktime(tmp) * 1e9
        # if are just processing child rows, then the name is already known
        out[self.col_name] = pd.to_datetime(col)
        out = out.apply(self.get_val, axis=1)

        # Handle missing
        if missing:
            nt = NullTransformer()
            res = nt.fit_transform(out, col_meta)
            return res
        return out.to_frame(self.col_name)

    def reverse_transform(self, col, col_meta, missing=True):
        """ Converts data back into original format """
        output = pd.DataFrame()
        date_format = col_meta['format']
        col_name = col_meta['name']
        fn = self.get_date_converter(col_name, date_format)

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
        """ Converts datetime to number """
        try:
            tmp = parser.parse(str(x[self.col_name])).timetuple()
            return time.mktime(tmp) * 1e9

        except Exception:
            # use default value
            return np.nan

    def get_date_converter(self, col, meta):
        '''Returns a converter that takes in an integer representing ms
           and turns it into a string date

        :param col: name of column
        :type col: str
        :param missing: true if column has NULL values
        :type missing: bool
        :param meta: type of column values
        :type meta: str

        :returns: function
        '''

        def safe_date(x):
            t = x[col]
            if np.isnan(t):
                t = self.default_val
            if np.isposinf(t):
                t = sys.maxsize
            elif np.isneginf(t):
                t = -sys.maxsize
            tmp = time.localtime(float(t) / 1e9)
            return time.strftime(meta, tmp)

        return safe_date
