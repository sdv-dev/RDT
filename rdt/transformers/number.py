import pandas as pd
import numpy as np

from rdt.transformers.null import NullTransformer


class NumericalTransformer(NullTransformer):
    """Transformer for numerical data."""

    def fit(self, data):
        if isinstance(data, np.ndarray):
            data = pd.Series(data)

        self.dtype = data.dtype

    def transform(self, data):
        extra_column = None

        if self.null_column and isinstance(data, pd.Series):
            extra_column = self._get_null_column(data.isnull())

        if isinstance(data, np.ndarray):
            if self.null_column:
                extra_column = self._get_null_column(~data.astype('bool'))
            data = pd.Series(data)

        transformed = data
        default = self._get_default(data)
        if default is not None:
            transformed = data.fillna(default)

        return transformed.to_numpy(), extra_column

    def reverse_transform(self, data):
        if not isinstance(data, np.ndarray):
            raise ValueError

        return data.astype(self.dtype)
