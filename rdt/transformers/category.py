import numpy as np
import pandas as pd
from faker import Faker
from scipy.stats import norm

from rdt.transformers.base import BaseTransformer
from rdt.transformers.null import NullTransformer


class CatTransformer(BaseTransformer):
    """Transformer for categorical data.

    Args:
        column_metadata(dict): Meta information of the column.
        missing(bool): Wheter or not handle missing values using NullTransformer.
        anonymize (bool): Wheter or not replace the values of col before generating the
                          categorical_map.

        category (str): The type of data to ask faker for when anonimizing.

    """

    def __init__(self, anonymize=False, category=None, *args, **kwargs):
        """Initialize transformer."""

        super().__init__(type='categorical', *args, **kwargs)

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

    def anonymize_column(self, col, column_metadata):
        """Map the values of column to new ones of the same type.

        It replaces the values from others generated using `faker`. It will however,
        keep the original distribution. That mean that the generated `probability_map` for both
        will have the same values, but different keys.

        Args:
            col (pandas.DataFrame): Dataframe containing the column to anonymize.
            column_metadata (dict): Meta information of the column.

        Returns:
            pd.DataFrame: DataFrame with its values mapped to new ones,
                          keeping the original distribution.

        Raises:
            ValueError: A `ValueError` is raised if faker is not able to provide enought
                        different values.
        """

        column_metadata = column_metadata or self.column_metadata
        self.check_data_type(column_metadata)
        self.col_name = column_metadata['name']
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

    def fit(self, col, column_metadata=None, missing=None):
        """Prepare the transfomer to process data.

        This method exist only to enforce the usage of `fit_transform` if `anonymize` is True.

        Args:
            col(pandas.DataFrame): Data to transform.
            column_metadata(dict): Meta information of the column.
            missing(bool): Wheter or not handle missing values using NullTransformer.

        Raises:
            ValueError: If this method is called and `anonymize` is True.

        """

        if self.anonymize:
            raise ValueError(
                '`fit` method is disabled when `anonymize` is True, '
                'please use fit_transform instead'
            )
        self._fit(col, column_metadata, missing)

    def _fit(self, col, column_metadata=None, missing=None):
        """Prepare the transfomer to process data.

        Args:
            col(pandas.DataFrame): Data to transform.
            column_metadata(dict): Meta information of the column.
            missing(bool): Wheter or not handle missing values using NullTransformer.
        """

        column_metadata = column_metadata or self.column_metadata
        self.check_data_type(column_metadata)
        self.col_name = column_metadata['name']

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

    def transform(self, col, column_metadata=None, missing=None):
        """Prepare the transformer to convert data and return the processed table.

        Args:
            col(pandas.DataFrame): Data to transform.
            column_metadata(dict): Meta information of the column.
            missing(bool): Wheter or not handle missing values using NullTransformer.

        Returns:
            pandas.DataFrame
        """

        column_metadata = column_metadata or self.column_metadata
        missing = missing if missing is not None else self.missing

        self.check_data_type(column_metadata)

        out = pd.DataFrame()

        # Make sure all nans are handled the same by replacing with None
        column = col[self.col_name].replace({np.nan: None})
        out[self.col_name] = column.apply(self.get_val)
        # Handle missing

        if missing:
            nt = NullTransformer()
            res = nt.fit_transform(out, column_metadata)
            return res

        return out

    def fit_transform(self, col, column_metadata=None, missing=None):
        """Prepare the transformer and return processed data.

        Args:
            col(pandas.DataFrame): Data to transform.
            column_metadata(dict): Meta information of the column.
            missing(bool): Wheter or not handle missing values using NullTransformer.

        Returns:
            pandas.DataFrame
        """

        if self.anonymize:
            col = self.anonymize_column(col, column_metadata)

        self._fit(col, column_metadata, missing)
        return self.transform(col, column_metadata, missing)

    def reverse_transform(self, col, column_metadata=None, missing=None):
        """Converts data back into original format.

        Args:
            col(pandas.DataFrame): Data to transform.
            column_metadata(dict): Meta information of the column.
            missing(bool): Wheter or not handle missing values using NullTransformer.

        Returns:
            pandas.DataFrame
        """
        column_metadata = column_metadata or self.column_metadata
        missing = missing if missing is not None else self.missing

        self.check_data_type(column_metadata)

        output = pd.DataFrame()
        new_col = self.get_category(col[self.col_name])

        if missing:
            new_col = new_col.rename(self.col_name)
            data = pd.concat([new_col, col['?' + self.col_name]], axis=1)
            nt = NullTransformer()
            output[self.col_name] = nt.reverse_transform(data, column_metadata)

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
