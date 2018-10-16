import sys

import numpy as np
import pandas as pd

from rdt.transformers.BaseTransformer import BaseTransformer
from rdt.transformers.NullTransformer import NullTransformer


class NumberTransformer(BaseTransformer):
    """Transformer for numerical data."""

    def __init__(self, *args, **kwargs):
        """Initialize transformer."""
        super().__init__(type='number', *args, **kwargs)
        self.default_val = None
        self.subtype = None

    def fit_transform(self, col, col_meta=None, missing=None):
        """Prepare the transformer to convert data and return the processed table.

        Args:
            col(pandas.DataFrame): Data to transform.
            col_meta(dict): Meta information of the column.
            missing(bool): Wheter or not handle missing values using NullTransformer.

        Returns:
            pandas.DataFrame
        """

        col_meta = col_meta or self.col_meta
        missing = missing if missing is not None else self.missing

        self.check_data_type(col_meta)

        out = pd.DataFrame()
        self.col_name = col_meta['name']
        self.subtype = col_meta['subtype']
        self.default_val = self.get_default_value(col)

        # if are just processing child rows, then the name is already known
        out[self.col_name] = col

        # Handle missing
        if missing:
            nt = NullTransformer()
            out = nt.fit_transform(out, col_meta)
            out[self.col_name] = out.apply(self.get_val, axis=1)
            return out

        out = out.apply(self.get_val, axis=1)

        if self.subtype == 'int':
            out[self.col_name] = out[self.col_name].astype(int)

        return out.to_frame(self.col_name)

    def reverse_transform(self, col, col_meta=None, missing=None):
        """Converts data back into original format.

        Args:
            col(pandas.DataFrame): Data to transform.
            col_meta(dict): Meta information of the column.
            missing(bool): Wheter or not handle missing values using NullTransformer.

        Returns:
            pandas.DataFrame
        """

        col_meta = col_meta or self.col_meta
        missing = missing if missing is not None else self.missing

        self.check_data_type(col_meta)

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
            output[col_name] = col.apply(fn, axis=1)

        if self.subtype == 'int':
            output[self.col_name] = output[self.col_name].astype(int)

        return output

    def get_default_value(self, data):
        """ """
        col = data[self.col_name]
        value = col[~col.isnull()].unique()[0]

        if self.subtype == 'integer':
            value = int(value)

        return value

    def get_val(self, x):
        """Converts to int."""
        try:
            if self.subtype == 'integer':
                return int(round(x[self.col_name]))
            else:
                if np.isnan(x[self.col_name]):
                    return self.default_val

                return x[self.col_name]

        except (ValueError, TypeError):
            return self.default_val

    def get_number_converter(self, col_name, subtype):
        """Returns a converter that takes in a value and turns it into an integer, if necessary.

        Args:
            col_name(str): Name of the column.
            subtype(str): Numeric subtype of the values.

        Returns:
            function
        """

        def safe_round(x):
            val = x[col_name]
            if np.isposinf(val):
                val = sys.maxsize
            elif np.isneginf(val):
                val = -sys.maxsize
            if np.isnan(val):
                val = self.default_val
            if subtype == 'integer':
                return int(round(val))
            return val

        return safe_round
