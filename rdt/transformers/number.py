import pandas as pd
import numpy as np

from rdt.transformers.base import BaseTransformer


class NumericalTransformer(BaseTransformer):
    """Transformer for numerical data."""

    default = None

    def __init__(self, settings={}):
        self.nan = settings.get('nan', 'mean')
        self.null_column = settings.get('null_column', True)
        self.default = None

    def _get_default(self, data):
        if self.nan == 'ignore':
            return None

        if self.nan == 'mean':
            _slide = ~data.isnull()
            return data[_slide].mean()

        if self.nan == 'mode':
            data_mode = data.mode()
            return data_mode[data_mode.first_valid_index()]

        return self.nan

    def fit(self, data):
        if isinstance(data, np.ndarray):
            data = pd.Series(data)

        self.dtype = data.dtype
        self.default = self._get_default(data)

    def _get_null_column(self, data):
        vfunc = np.vectorize(lambda x: 1 if x and not np.isnan(x) else 0)
        return vfunc(data)

    def transform(self, data):
        extra_column = None

        if self.null_column and isinstance(data, pd.Series):
            extra_column = self._get_null_column(data.to_numpy())

        if isinstance(data, np.ndarray):
            if self.null_column:
                extra_column = self._get_null_column(data)
            data = pd.Series(data)

        transformed = data
        if self.default is not None:
            transformed = data.fillna(self.default)

        return transformed.to_numpy(), extra_column

    def reverse_transform(self, data):
        if not isinstance(data, np.ndarray):
            raise ValueError

        return data.astype(self.dtype)
