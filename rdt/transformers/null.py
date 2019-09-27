import numpy as np

from rdt.transformers.base import BaseTransformer


class NullTransformer(BaseTransformer):
    """Transformer for null data."""

    def __init__(self, **kwargs):
        self.nan = kwargs.get('nan', 'mean')
        self.null_column = kwargs.get('null_column', True)

    def _get_null_column(self, data):
        """Get null column with 0 or 1 values.

        Args:
            data (numpy.ndarray):
                Data used to generate a new column.

        Returns:
            numpy.ndarray
        """
        vfunc = np.vectorize(lambda x: 1 if x is True else 0)
        return vfunc(data)

    def _get_default(self, data):
        """Get the value to replace null values in a column.

        Args:
            data (pandas.Series):
                Data used to compute the default value.
        """
        if self.nan == 'ignore':
            return None

        if self.nan == 'mean':
            _slide = ~data.isnull()
            return data[_slide].mean()

        if self.nan == 'mode':
            data_mode = data.mode()
            return data_mode[data_mode.first_valid_index()]

        return self.nan
