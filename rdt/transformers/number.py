import sys

import numpy as np
import pandas as pd

from rdt.transformers.base import BaseTransformer


class NumberTransformer(BaseTransformer):
    """Transformer for numerical data."""

    type = 'number'

    def __init__(self, *args, **kwargs):
        """Initialize transformer."""
        super().__init__(*args, **kwargs)
        self.subtype = self.column_metadata['subtype']
        self.default_val = None

    def fit(self, col):
        """Sets the default value.

        Args:
            col (pandas.DataFrame):
                Data to transform.

        Returns:
            pandas.DataFrame
        """
        self.default_val = self.get_default_value(col)

    def transform(self, col):
        """Prepare the transformer to convert data and return the processed table.

        Args:
            col (pandas.DataFrame):
                Data to transform.

        Returns:
            pandas.DataFrame
        """

        out = pd.DataFrame(index=col.index)
        out[self.col_name] = col.apply(self.get_val, axis=1)

        if self.subtype == 'int':
            out[self.col_name] = out[self.col_name].astype(int)

        return out

    def reverse_transform(self, col):
        """Converts data back into original format.

        Args:
            col (pandas.DataFrame):
                Data to transform.

        Returns:
            pandas.DataFrame
        """

        output = pd.DataFrame(index=col.index)
        output[self.col_name] = col.apply(self.safe_round, axis=1)

        if self.subtype == 'int':
            output[self.col_name] = output[self.col_name].astype(int)

        return output

    def get_default_value(self, data):
        col = data[self.col_name]
        uniques = col[~col.isnull()].unique()
        if not len(uniques):
            value = 0

        else:
            value = uniques[0]

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

    def safe_round(self, x):
        """Returns a converter that takes in a value and turns it into an integer, if necessary.

        Args:
            col_name (str):
                Name of the column.
            subtype (str):
                Numeric subtype of the values.

        Returns:
            int
        """
        val = x[self.col_name]

        if np.isposinf(val):
            val = sys.maxsize

        elif np.isneginf(val):
            val = -sys.maxsize

        if np.isnan(val):
            val = self.default_val

        if self.subtype == 'integer':
            return int(round(val))

        return val
