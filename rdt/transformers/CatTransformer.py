import numpy as np
import pandas as pd
from faker import Faker
from scipy.stats import norm

from rdt.transformers.BaseTransformer import BaseTransformer
from rdt.transformers.NullTransformer import NullTransformer


class CatTransformer(BaseTransformer):
    """Transformer for categorical data.

    Args:
        col_meta(dict): Meta information of the column.
        missing(bool): Wheter or not handle missing values using NullTransformer.
        anonimize (bool): Wheter or not replace the values of col before generating the
                          categorical_map.

        category (str): The type of data to ask faker for when anonimizing.

    """
    Faker = Faker

    def __init__(self, anonimize=False, category=None, *args, **kwargs):
        """Initialize transformer."""
        super().__init__(type='categorical', *args, **kwargs)

        if anonimize and not category:
            raise ValueError('`category` must be specified if `anonimize` is True')

        self.anonimize = anonimize
        self.category = category
        self.probability_map = {}

        if self.anonimize and self.category:
            self.get_generator()

    def get_generator(self):
        """Return the generator object to anonimize data."""
        faker = self.Faker()

        try:
            return getattr(faker, self.category)

        except AttributeError as e:
            raise ValueError('Category {} couldn\'t be found on faker')

    def anonimize_column(self, col, col_meta):
        """ """

        col_meta = col_meta or self.col_meta
        self.check_data_type(col_meta)
        self.col_name = col_meta['name']
        column = col[self.col_name]

        generator = self.get_generator()
        original_values = column[~pd.isnull(column)].unique()
        new_values = [generator() for x in range(len(original_values))]

        if len(new_values) != len(set(new_values)):
            raise ValueError(
                'There are not enought different values on faker provider'
                'for category {}'.format(self.category)
            )

        value_map = dict(zip(original_values, new_values))
        column = column.apply(value_map.get)

        return column.to_frame()

    def fit(self, col, col_meta=None, missing=None):
        """Maps each unique value to probability of seeing it."""

        if self.anonimize:
            raise ValueError(
                '`fit` method is disabled when `anonimize` is True, '
                'please use fit_transform instead'
            )
        self._fit(col, col_meta, missing)

    def _fit(self, col, col_meta=None, missing=None):
        """ """
        col_meta = col_meta or self.col_meta
        self.check_data_type(col_meta)
        self.col_name = col_meta['name']

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

    def transform(self, col, col_meta=None, missing=None):
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

        # Make sure all nans are handled the same by replacing with None
        column = col[self.col_name].replace({np.nan: None})
        out[self.col_name] = column.apply(self.get_val)
        # Handle missing

        if missing:
            nt = NullTransformer()
            res = nt.fit_transform(out, col_meta)
            return res

        return out

    def fit_transform(self, col, col_meta=None, missing=None):
        """Prepare the transformer to convert data and return the processed table.

        Args:
            col(pandas.DataFrame): Data to transform.
            col_meta(dict): Meta information of the column.
            missing(bool): Wheter or not handle missing values using NullTransformer.


        Returns:
            pandas.DataFrame
        """
        if self.anonimize:
            col = self.anonimize_column(col, col_meta)

        self._fit(col, col_meta, missing)
        return self.transform(col, col_meta, missing)

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
        new_col = self.get_category(col[self.col_name])

        if missing:
            new_col = new_col.rename(self.col_name)
            data = pd.concat([new_col, col['?' + self.col_name]], axis=1)
            nt = NullTransformer()
            output[self.col_name] = nt.reverse_transform(data, col_meta)

        else:
            output[self.col_name] = new_col

        return output

    def get_val(self, x):
        """Convert cat value into num between 0 and 1."""
        interval, mean, std = self.probability_map[x]
        new_val = norm.rvs(mean, std)
        return new_val

    def get_category(self, column):
        """Returns categories for the specified numeric values

        Args:
            column(pandas.Series): Values to transform into categories

        Returns:
            pandas.Series
        """
        result = pd.Series(index=column.index)

        for category, stats in self.probability_map.items():
            start, end = stats[0]
            result[(start < column) & (column < end)] = category

        return result
