import time
import numpy as np
import pandas as pd

from rdt.transformers.BaseTransformer import *
from dateutil import parser


class DTTransformer(BaseTransformer):
    """
    This class represents the datetime transformer for SDV
    """

    def __init__(self):
        """ initialize transformer """
        super(DTTransformer, self).__init__()
        self.type = 'datetime'
        self.col_name = None
        self.default_val = None

    def fit_transform(self, col, col_meta):
        """ Returns a tuple (transformed_table, new_table_meta) """
        out = pd.DataFrame(columns=[])
        self.col_name = col_meta['name']
        # get default val
        for x in col:
            if x is not None:
                try:
                    tmp = parser.parse(str(x)).timetuple()
                    self.default_val = time.mktime(tmp)*1e9
                    break
                except Exception as inst:
                    continue
        # if are just processing child rows, then the name is already known
        out[self.col_name] = col
        out = out.apply(self.get_val, axis=1)
        return out.to_frame(self.col_name)

    def transform(self, col, col_meta):
        """ Does the required transformations to the data """
        return self.fit_transform(col, col_meta)

    def reverse_transform(self, col, col_meta):
        """ Converts data back into original format """
        output = pd.DataFrame(columns=[])
        date_format = col_meta['format']
        col_name = col_meta['name']
        fn = self.get_date_converter(col_name, date_format)
        data = col.to_frame()
        output[col_name] = data.apply(fn, axis=1)
        return output

    def get_val(self, x):
        """ Converts datetime to number """
        try:
            tmp = parser.parse(str(x[self.col_name])).timetuple()
            return time.mktime(tmp)*1e9
        except Exception:
            # use default value
            return self.default_val

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
            tmp = time.gmtime(float(t)/1e9)
            return time.strftime(meta, tmp)

        return safe_date
