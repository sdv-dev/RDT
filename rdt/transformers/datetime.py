import time

import numpy as np
import pandas as pd

from rdt.transformers.base import BaseTransformer
from rdt.transformers.null import NullTransformer


class DateTimeTransformer(BaseTransformer):
    """Transformer for datetime data."""

    def __init__(self, **kwargs):
        self.datetime_format = kwargs.get('format')
        self.nan = kwargs.get('nan', 'mean')

        if kwargs.get('null_column', True):
            self.null_transformer = NullTransformer()
        else:
            self.null_transformer = None

    @classmethod
    def _get_null_column(cls, data):
        vfunc = np.vectorize(lambda x: 1 if x is True else 0)

        return vfunc(data)

    def _get_default(self, data):
        if self.nan == 'ignore':
            return None

        if self.nan == 'mean':
            _slice = ~data.isnull()
            return data[_slice].mean()

        if self.nan == 'mode':
            data_mode = data.mode(dropna=True)
            return data_mode.iloc[0]

        return self.nan

    def transform(self, data):
        if isinstance(data, np.ndarray):
            data = pd.Series(data)

        if self.null_transformer is not None:
            data = self.null_transformer.fit_transform(data)

        else:
            data = data.to_frame()

        data.iloc[:,0] = pd.to_datetime(data.iloc[:,0], format=self.datetime_format, errors='coerce')

        default = self._get_default(data.iloc[:,0])
        if default is not None:
            default = pd.to_datetime(default, format=self.datetime_format, errors='coerce')
            data.iloc[:,0] = data.iloc[:,0].fillna(default)

        data.iloc[:,0] = data.iloc[:,0].astype('int64')

        return data

    def _transform_to_date(self, data):
        """Transform a numeric value into str datetime format."""
        aux_time = time.localtime(float(data) / 1e9)

        return time.strftime(self.datetime_format, aux_time)

    def reverse_transform(self, data):
        vect_func = np.vectorize(self._transform_to_date)

        return vect_func(data)
