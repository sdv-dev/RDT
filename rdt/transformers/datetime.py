import time

import numpy as np
import pandas as pd

from rdt.transformers.base import BaseTransformer
from rdt.transformers.null import NullTransformer


class DateTimeTransformer(BaseTransformer):
    """Transformer for datetime data."""

    def __init__(self, format, nan='mean', null_column=True):
        self.datetime_format = format
        self.nan = nan
        self.null_column = null_column
        self.null_transformer = None

    def fit(self, data):
        if isinstance(data, np.ndarray):
            data = pd.Series(data)

        transformed = pd.to_datetime(data, format=self.datetime_format, errors='coerce')
        transformed = transformed.astype('int64')

        if self.nan == 'mean':
            fill_value = transformed.mean()
        elif self.nan == 'mode':
            fill_value = transformed.mode(dropna=True)[0]
        elif self.nan == 'ignore':
            fill_value = None
        else:
            fill_value = self.nan

        if fill_value is None:
            self.null_transformer = NullTransformer(fill_value, self.null_column)

        else:
            fill_value = pd.to_datetime(fill_value, format=self.datetime_format, errors='coerce')
            fill_value = fill_value.strftime(self.datetime_format)
            self.null_transformer = NullTransformer(fill_value, self.null_column)

    def transform(self, data):
        if isinstance(data, np.ndarray):
            data = pd.Series(data)

        transformed = self.null_transformer.transform(data)
        shape = transformed.shape

        if self.null_column and len(shape) == 2 and shape[1] == 2:
            _slice = transformed[:, 0].astype('bool')

            transformed[_slice, 0] = pd.to_datetime(
                transformed[_slice, 0],
                format=self.datetime_format,
                errors='coerce'
            ).astype('int64').to_numpy()

            return transformed

        transformed = pd.to_datetime(
            transformed,
            format=self.datetime_format,
            errors='coerce'
        )

        return transformed.astype('int64').to_numpy()

    def _transform_to_date(self, data):
        """Transform a numeric value into str datetime format."""
        if data is None:
            return None

        aux_time = time.localtime(float(data) / 1e9)

        return time.strftime(self.datetime_format, aux_time)

    def reverse_transform(self, data):
        vect_func = np.vectorize(self._transform_to_date)

        return vect_func(data)
