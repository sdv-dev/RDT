import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

from rdt.transformers.base import BaseTransformer


class DTTransformer(BaseTransformer):
    """Transformer for datetime data."""

    type = 'datetime'

    def __init__(self, *args, **kwargs):
        """Initialize transformer."""
        super().__init__(*args, **kwargs)

        self.default_val = None
        self.date_format = self.column_metadata['format']

    def fit(self, col):
        """Prepare the transformer to convert data.

        Args:
            col (pandas.DataFrame):
                Data to transform.
        """
        dates = self.safe_datetime_cast(col)
        self.default_val = dates.groupby(dates).count().index[0].timestamp() * 1e9

    def transform(self, col):
        """Prepare the transformer to convert data and return the processed table.

        Args:
            col (pandas.DataFrame):
                Data to transform.

        Returns:
            pandas.DataFrame
        """
        out = pd.DataFrame()
        out[self.col_name] = self.safe_datetime_cast(col)
        out[self.col_name] = self.to_timestamp(out)

        return out

    def strptime_format(self, x):
        return datetime.strptime(x, self.date_format)

    def reverse_transform(self, col):
        """Converts data back into original format.

        Args:
            col (pandas.DataFrame):
                Data to transform.

        Returns:
            pandas.DataFrame
        """
        if isinstance(col, pd.Series):
            col = col.to_frame()

        output = pd.DataFrame(index=col.index)
        output[self.col_name] = col.apply(self.safe_date, axis=1)

        return output

    def safe_datetime_cast(self, col):
        """Parses string values into datetime.

        Args:
            col (pandas.DataFrame):
                Data to transform.

        Returns:
            pandas.Series
        """
        casted_dates = pd.to_datetime(col[self.col_name], format=self.date_format, errors='coerce')

        if len(casted_dates[casted_dates.isnull()]):
            # This will raise an error for bad formatted data
            # but not for out of bonds or missing dates.
            slice_ = casted_dates.isnull() & ~col[self.col_name].isnull()
            col[slice_][self.col_name].apply(self.strptime_format)

        return casted_dates

    def to_timestamp(self, data):
        """Transform a datetime series into linux epoch.

        Args:
            data (pandas.DataFrame):
                DataFrame containins a column named as ``self.col_name``.

        Returns:
            pandas.Series
        """
        result = pd.Series(index=data.index)
        _slice = ~data[self.col_name].isnull()

        result[_slice] = data[_slice][self.col_name].astype('int64')
        return result

    def safe_date(self, x):
        """Transform ``x[self.col_name]`` into a date string.

        Args:
            x (dict-like or pandas.Series):
                Row containing data to cast safely.

        Returns:
            str
        """

        t = x[self.col_name]
        if np.isnan(t):
            return t

        elif np.isposinf(t):
            t = sys.maxsize

        elif np.isneginf(t):
            t = -sys.maxsize

        tmp = time.localtime(float(t) / 1e9)
        return time.strftime(self.date_format, tmp)
