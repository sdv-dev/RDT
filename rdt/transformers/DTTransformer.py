import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

from rdt.transformers.BaseTransformer import BaseTransformer
from rdt.transformers.NullTransformer import NullTransformer


class DTTransformer(BaseTransformer):
    """Transformer for datetime data."""

    def __init__(self, *args, **kwargs):
        """Initialize transformer."""
        super().__init__(type='datetime', *args, **kwargs)
        self.default_val = None

    def fit(self, col, col_meta=None, missing=None):
        """Prepare the transformer to convert data.

        Args:
            col(pandas.DataFrame): Data to transform.
            col_meta(dict): Meta information of the column.
            missing(bool): Wheter or not handle missing values using NullTransformer.

        Returns:
            None
        """
        col_meta = col_meta or self.col_meta
        missing = missing if missing is not None else self.missing

        self.check_data_type(col_meta)
        self.col_name = col_meta['name']

        # get default val
        dates = self.safe_datetime_cast(col, col_meta)
        self.default_val = dates.groupby(dates).count().index[0].timestamp() * 1e9

    def transform(self, col, col_meta=None, missing=None):
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
        out[self.col_name] = self.safe_datetime_cast(col, col_meta)
        out[self.col_name] = self.to_timestamp(out)

        # Handle missing
        if missing:
            nt = NullTransformer()
            res = nt.fit_transform(out, col_meta)
            return res

        return out

    @staticmethod
    def strptime_format(date_format):
        def f(x):
            return datetime.strptime(x, date_format)

        return f

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

        output = pd.DataFrame()
        date_format = col_meta['format']

        fn = self.get_date_converter(self.col_name, date_format)
        reversed_column = col.apply(fn, axis=1)

        if missing:
            reversed_column = reversed_column.rename(self.col_name)
            data = pd.concat([reversed_column, col['?' + self.col_name]], axis=1)
            nt = NullTransformer()
            output[self.col_name] = nt.reverse_transform(data, col_meta)

        else:
            output[self.col_name] = reversed_column

        return output

    def safe_datetime_cast(self, col, col_meta):
        """Parses string values into datetime.

        Args:
            col(pandas.DataFrame): Data to transform.
            col_meta(dict): Meta information of the column.

        Returns:
            pandas.Series
        """
        date_format = col_meta['format']
        casted_dates = pd.to_datetime(col[self.col_name], format=date_format, errors='coerce')

        if len(casted_dates[casted_dates.isnull()]):
            # This will raise an error for bad formatted data
            # but not for out of bonds or missing dates.
            slice_ = casted_dates.isnull() & ~col[self.col_name].isnull()
            col[slice_][self.col_name].apply(self.strptime_format(date_format))

        return casted_dates

    def to_timestamp(self, data):
        """Transform a datetime series into linux epoch.

        Args:
            data(pandas.DataFrame): DataFrame containins a column named as `self.col_name`.

        Returns:
            pandas.Series
        """
        result = pd.Series(index=data.index)
        _slice = ~data[self.col_name].isnull()

        result[_slice] = data[_slice][self.col_name].apply(datetime.timestamp) * 1e9
        return result

    def get_date_converter(self, col_name, date_format):
        """Return a function that takes in an integer representing ms and return a string date.

        Args:
            col_name(str): Name of the column.
            missing(bool): Wheter or not the column has null values.
            date_format(str): Output date in strftime format.

        Returns:
            function
        """

        def safe_date(x):
            t = x[col_name]
            if np.isnan(t):
                t = self.default_val

            elif np.isposinf(t):
                t = sys.maxsize

            elif np.isneginf(t):
                t = -sys.maxsize

            tmp = time.localtime(float(t) / 1e9)
            return time.strftime(date_format, tmp)

        return safe_date
