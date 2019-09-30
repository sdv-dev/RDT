import time

import numpy as np
import pandas as pd

from rdt.transformers.null import NullTransformer


class DateTimeTransformer(NullTransformer):
    """Transformer for datetime data."""

    def __init__(self, **kwargs):
        super(DateTimeTransformer, self).__init__(**kwargs)
        self.datetime_format = kwargs.get('format')

    def transform(self, data):
        if isinstance(data, np.ndarray):
            data = pd.Series(data)

        extra_column = None
        if self.null_column:
            extra_column = self._get_null_column(data.isnull())

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
