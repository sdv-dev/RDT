import numpy as np
import pandas as pd

from rdt.transformers.base import BaseTransformer


class PositiveNumberTransformer(BaseTransformer):

    type = 'number'

    def fit(self, column):
        pass

    def transform(self, column):
        """Applies an exponential to values to turn them positive numbers.

        Args:
            column (pandas.DataFrame):
                Data to transform.

        Returns:
            pandas.DataFrame
        """
        self.check_data_type()

        return pd.DataFrame({self.col_name: np.exp(column[self.col_name])})

    def reverse_transform(self, column):
        """Applies the natural logarithm function to turn positive values into real ranged values.

        Args:
            column (pandas.DataFrame):
                Data to transform.

        Returns:
            pandas.DataFrame
        """
        self.check_data_type()

        return pd.DataFrame({self.col_name: np.log(column[self.col_name])})
