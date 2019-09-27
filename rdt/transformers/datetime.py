import time

import numpy as np
import pandas as pd

from rdt.transformers.base import BaseTransformer


class DateTimeTransformer(BaseTransformer):
    """Transformer for datetime data."""

    def __init__(self, settings={}):
        self.datetime_format = settings['datetime_format']
        self.nan = settings.get('nan', 'mean')
        self.null_column = settings.get('null_column', True)

    def _get_default(self, data):
        if isinstance(data, np.ndarray):
            data = pd.Series(data)

        if self.nan == 'ignore':
            return None

        if self.nan == 'mean':
            _slide = ~data.isnull()
            return data[_slide].mean()

        if self.nan == 'mode':
            data_mode = pd.to_datetime(data).mode()
            return data_mode[data_mode.first_valid_index()]

        return self.nan

    def _get_null_column(self, data):
        vfunc = np.vectorize(lambda x: 1 if x else 0)
        return vfunc(data)

    def transform(self, data):
        if isinstance(data, np.ndarray):
            data = pd.Series(data)

        extra_column = None
        if self.null_column:
            extra_column = self._get_null_column(data.to_numpy())

        data = pd.to_datetime(data, format=self.datetime_format, errors='coerce')

        default = self._get_default(data)
        if default is not None:
            default = pd.to_datetime(default, format=self.datetime_format, errors='coerce')
            data = data.fillna(default)
        
        return data.to_numpy().astype('int64'), extra_column

    def _transform_to_date(self, data):
        """Transform a numeric value into str datetime format."""
        aux_time = time.localtime(float(data) / 1e9)

        return time.strftime(self.datetime_format, aux_time)

    def reverse_transform(self, data):
        vect_func = np.vectorize(self._transform_to_date)

        return vect_func(data)
