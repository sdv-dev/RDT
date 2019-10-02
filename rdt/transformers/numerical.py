import numpy as np
import pandas as pd

from rdt.transformers.base import BaseTransformer
from rdt.transformers.null import NullTransformer


class NumericalTransformer(BaseTransformer):
    """Transformer for numerical data."""

    def __init__(self, **kwargs):
        self.nan = kwargs.get('nan', 'mean')
        self.dtype = None

        if kwargs.get('null_column', True):
            self.null_transformer = NullTransformer()

        else:
            self.null_transformer = None

    def fit(self, data):
        if isinstance(data, np.ndarray):
            data = pd.Series(data)

        self.dtype = data.dtype

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

        default = self._get_default(data[0])
        if default is not None:
            data[0] = data[0].fillna(default)

        return data

    def reverse_transform(self, data):
        return data.astype(self.dtype)
