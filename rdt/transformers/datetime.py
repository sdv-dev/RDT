import time

import numpy as np
import pandas as pd

from rdt.transformers.base import BaseTransformer


class DateTimeTransformer(BaseTransformer):
    """Transformer for datetime data."""

    def __init__(self, datetime_format):
        self.datetime_format = datetime_format

    def transform(self, data):
        if isinstance(data, pd.Series):
            data = data.to_numpy()

        datetime_data = pd.to_datetime(data, format=self.datetime_format, errors='coerce')

        # Don't transform NaT
        # REVIEW
        _slice = ~datetime_data.isnull()
        transformed = datetime_data[_slice].to_numpy().astype('int64')

        return transformed

    def _transform_to_date(self, data):
        """Transform a numeric value into str datetime format."""
        aux_time = time.localtime(float(data) / 1e9)

        return time.strftime(self.datetime_format, aux_time)

    def reverse_transform(self, data):
        vect_func = np.vectorize(self._transform_to_date)

        return vect_func(data)
