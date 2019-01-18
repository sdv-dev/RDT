import numpy as np
import pandas as pd
from faker import Faker
from scipy.stats import norm

from rdt.transformers.base import BaseTransformer


class CatTransformer(BaseTransformer):
    """Transformer for categorical data.

    Args:
        column_metadata(dict): Meta information of the column.
        anonymize (bool): Wheter or not replace the values of col before generating the
                          categorical_map.

        category (str): The type of data to ask faker for when anonimizing.

    """

    type = 'categorical'

    def __init__(self, column_metadata, anonymize=False, category=None):
        """Initialize transformer."""

        super().__init__(column_metadata)

        if anonymize and not category:
            raise ValueError('`category` must be specified if `anonymize` is True')

        self.anonymize = anonymize
        self.category = category
        self.probability_map = {}

        if self.anonymize and self.category:
            self.get_generator()

    def get_generator(self):
        """Return the generator object to anonymize data."""

        faker = Faker()

        try:
            return getattr(faker, self.category)

        except AttributeError:
            raise ValueError('Category {} couldn\'t be found on faker')

    def anonymize_column(self, col):
        """Map the values of column to new ones of the same type.

        It replaces the values from others generated using `faker`. It will however,
        keep the original distribution. That mean that the generated `probability_map` for both
        will have the same values, but different keys.

        Args:
            col (pandas.DataFrame): Dataframe containing the column to anonymize.

        Returns:
            pd.DataFrame: DataFrame with its values mapped to new ones,
                          keeping the original distribution.

        Raises:
            ValueError: A `ValueError` is raised if faker is not able to provide enought
                        different values.
        """

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

    def fit(self, col):
        """Prepare the transfomer to process data.

        This method can only be used if `anonymize` is False.
        Otherwise, please use `fit_transform`.

        Args:
            col(pandas.DataFrame): Data to transform.

        Raises:
            ValueError: If this method is called and `anonymize` is True.

        """

        if self.anonymize:
            raise ValueError(
                '`fit` method is disabled when `anonymize` is True, '
                'please use fit_transform instead'
            )
        self._fit(col)

    def _fit(self, col):
        """Create a map of the empirical probability for each category.

        Args:
            col(pandas.DataFrame): Data to transform.
        """

        column = col[self.col_name].replace({np.nan: np.inf})
        frequencies = column.groupby(column).count().rename({np.inf: None}).to_dict()
        # next set probability ranges on interval [0,1]
        start = 0
        end = 0
        num_vals = len(col)

        for val in frequencies:
            prob = frequencies[val] / num_vals
            end = start + prob
            interval = (start, end)
            mean = np.mean(interval)
            std = prob / 6
            self.probability_map[val] = (interval, mean, std)
            start = end

    def transform(self, col):
        """Prepare the transformer to convert data and return the processed table.

        Args:
            col(pandas.DataFrame): Data to transform.

        Returns:
            pandas.DataFrame
        """

        out = pd.DataFrame()

        # Make sure all nans are handled the same by replacing with None
        column = col[self.col_name].replace({np.nan: None})
        out[self.col_name] = column.apply(self.get_val)

        return out

    def fit_transform(self, col):
        """Prepare the transformer and return processed data.

        Args:
            col(pandas.DataFrame): Data to transform.

        Returns:
            pandas.DataFrame
        """

        if self.anonymize:
            col = self.anonymize_column(col)

        self._fit(col)
        return self.transform(col)

    def reverse_transform(self, col, column_metadata=None):
        """Converts data back into original format.

        Args:
            col(pandas.DataFrame): Data to transform.

        Returns:
            pandas.DataFrame
        """

        output = pd.DataFrame()
        new_col = self.get_category(col[self.col_name])
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
