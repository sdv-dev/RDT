import numpy as np
import pandas as pd
from scipy.stats import norm

from rdt.transformers.BaseTransformer import BaseTransformer
from rdt.transformers.NullTransformer import NullTransformer


class CatTransformer(BaseTransformer):
    """Transformer for categorical data."""

    def __init__(self, *args, **kwargs):
        """Initialize transformer."""
        super().__init__(type='categorical', *args, **kwargs)
        self.probability_map = {}  # val -> ((a,b), mean, std)

    def fit_transform(self, col, col_meta=None, missing=None):
        """Prepare the transformer to convert data and return the processed table.

        Args:
            col(pandas.DataFrame): Data to transform.
            col_meta(dict): Meta information of the column.
            missing(bool): Wheter or not handle missing values using NullTransformer.

        Returns:
            pandas.DataFrame
        """

        col_meta = col_meta or self.col_meta
        missing = missing if missing is not None else self.missing

        self.check_data_type(col_meta)

        out = pd.DataFrame()
        self.col_name = col_meta['name']
        self.get_probability_map(col)

        # Make sure all nans are handled the same by replacing with None
        column = col[self.col_name].replace({np.nan: None})
        out[self.col_name] = column.apply(self.get_val)
        # Handle missing

        if missing:
            nt = NullTransformer()
            res = nt.fit_transform(out, col_meta)
            return res

        return out

    def reverse_transform(self, col, col_meta=None, missing=None):
        """Converts data back into original format.

        Args:
            col(pandas.DataFrame): Data to transform.
            col_meta(dict): Meta information of the column.
            missing(bool): Wheter or not handle missing values using NullTransformer.

        Returns:
            pandas.DataFrame
        """

        col_meta = col_meta or self.col_meta
        missing = missing if missing is not None else self.missing

        self.check_data_type(col_meta)

        output = pd.DataFrame()
        col_name = col_meta['name']
        fn = self.get_reverse_cat(col_name)
        new_col = col.apply(fn, axis=1)

        if missing:
            new_col = new_col.rename(col_name)
            data = pd.concat([new_col, col['?' + col_name]], axis=1)
            nt = NullTransformer()
            output[col_name] = nt.reverse_transform(data, col_meta)

        else:
            output[col_name] = new_col

        return output

    def get_val(self, x):
        """Convert cat value into num between 0 and 1."""
        interval, mean, std = self.probability_map[x]
        new_val = norm.rvs(mean, std)
        return new_val

    def get_reverse_cat(self, col_name):
        """Returns a converter that takes in a value and turns it back into a category."""

        def reverse_cat(x):
            res = None

            for val in self.probability_map:
                interval = self.probability_map[val][0]
                if x[col_name] >= interval[0] and x[col_name] < interval[1]:
                    return val

            if res is None:
                return list(self.probability_map.keys())[0]

        return reverse_cat

    def get_probability_map(self, col):
        """Maps each unique value to probability of seeing it."""
        column = col[self.col_name].replace({np.nan: np.inf})
        self.probability_map = column.groupby(column).count().rename({np.inf: None}).to_dict()
        # next set probability ranges on interval [0,1]
        cur = 0
        num_vals = len(col)
        for val in self.probability_map:
            prob = self.probability_map[val] / num_vals
            interval = (cur, cur + prob)
            cur = cur + prob
            mean = np.mean(interval)
            std = (interval[1] - interval[0]) / 6
            self.probability_map[val] = (interval, mean, std)
