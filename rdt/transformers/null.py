import pandas as pd
import numpy as np

from rdt.transformers.base import BaseTransformer


class NullTransformer(BaseTransformer):
    """Transformer for null data."""

    def transform(self, data):
        if isinstance(data, np.ndarray):
            data = pd.Series(data)

        vfunc = np.vectorize(lambda x: 1 if x is True else 0)
        null_column = vfunc(data.isnull())

        transformed = data.to_frame()
        transformed.insert(1, 1, null_column)

        return transformed

    def reverse_transform(self, data):
        pass
