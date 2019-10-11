import numpy as np
import pandas as pd

from rdt.transformers import (
    BooleanTransformer, CategoricalTransformer, DatetimeTransformer, NumericalTransformer,
    load_transformers)


class HyperTransformer:
    """HyperTransformer class.

    The ``HyperTransformer`` class contains a collection of ``transformers`` that can be
    used to transform and reverse transform one or more columns at once.

    Args:
        column_transformers (dict):
            Dict containing the name of the column as a key and the ``transformer`` instance as
            a value. Also the value can be a dictionary containing the information of the
            transformer, see the example below. Defaults to ``None``.
        copy (bool):
            Make a copy of the input data or not. Defaults to ``True``.
        anonymize (string, tuple or list):
            ``Faker`` method with arguments to be used in categoricals anonymization.
            Defaults to ``None``.
        dtypes (list):
            List of data types corresponding to ``column_transformers`` or the ``data``.
            List of ``dtype`` corresponding to the data columns to fit. Defaults to ``None``.

    Example:
        In this example we will create a ``HyperTransformer`` that will transform the
        columns ``a`` and ``b`` by passing it a ``dict`` containing the name of the column
        and the instance of the transformer to be used (for the column ``a``) and a ``dict``
        with the parameters for the column ``b`` containing the class name and the keyword
        arguments for another ``NumericalTransformer``.

        >>> nt = NumericalTransformer(dtype=float)
        >>> transformers = {
        ...     'a': nt,
        ...     'b': {
        ...         'class': 'NumericalTransformer',
        ...         'kwargs': {
        ...             'dtype': int
        ...         }
        ...     }
        ... }
        >>> ht = HyperTransformer(transformers)
    """
    def __init__(self, transformers=None, copy=True, anonymize=None, dtypes=None):
        self.transformers = transformers
        self._transformers = dict()
        self.copy = copy
        self.anonymize = anonymize or dict()
        self.dtypes = dtypes

    def _analyze(self, data):
        """Build a ``dict`` with column names and transformers from a given ``pandas.DataFrame``.

        When ``self.dtypes`` is ``None``, use the dtypes from the input data.

        When ``dtype`` is:
            - ``int``: a ``NumericalTransformer`` is created with ``dtype=int``.
            - ``float``: a ``NumericalTransformer`` is created with ``dtype=float``.
            - ``object``: a ``CategoricalTransformer`` is created.
            - ``bool``: a ``BooleanTransformer`` is created.
            - ``datetime64``: a ``DatetimeTransformer`` is created.

        Any other ``dtype`` is not supported and raise a ``ValueError``.

        Args:
            data (pandas.DataFrame):
                Data used to analyze the ``pandas.DataFrame`` dtypes.

        Returns:
            dict:
                Map column name with the created transformer.

        Raises:
            ValueError:
                A ``ValueError`` is raised if a ``dtype`` is not supported by the
               ``HyperTransformer``.
        """
        transformers = dict()
        dtypes = self.dtypes or data.dtypes
        for name, dtype in zip(data.columns, dtypes):
            if np.issubdtype(dtype, np.dtype(int)):
                transformer = NumericalTransformer(dtype=int)
            elif np.issubdtype(dtype, np.dtype(float)):
                transformer = NumericalTransformer(dtype=float)
            elif dtype == np.object:
                anonymize = self.anonymize.get(name)
                transformer = CategoricalTransformer(anonymize=anonymize)
            elif np.issubdtype(dtype, np.dtype(bool)):
                transformer = BooleanTransformer()
            elif np.issubdtype(dtype, np.datetime64):
                transformer = DatetimeTransformer()
            else:
                raise ValueError('Unsupported dtype: {}'.format(dtype))

            transformers[name] = transformer

        return transformers

    def fit(self, data):
        """Prepare transformers before convert data.

        Before fit the data, check if the transformers are already defined.
        If not, the transformers will be created analyzing the data types.

        Args:
            data (pandas.DataFrame):
                Data to fit.
        """
        if self.transformers:
            self._transformers = load_transformers(self.transformers)
        else:
            self._transformers = self._analyze(data)

        for column_name, transformer in self._transformers.items():
            column = data[column_name]
            transformer.fit(column)

    def transform(self, data):
        """Does the required transformations to the data.

        If ``self.copy`` is ``True`` make a copy of the data to don't overwrite it.

        When a ``NullTransformer`` is applied it can generate a new column with values in range
        1 or 0 if the values are null or not respectively.
        New columns will be named column_name#1.

        Also, a ``NullTransformer`` can replace null values. If this fill value is already
        in the data and we don't create a null column data can't be reversed. In this case
        we show a warning.

        Args:
            data (pandas.DataFrame):
                Data to transform.

        Returns:
            pandas.DataFrame
        """
        if self.copy:
            data = data.copy()

        for column_name, transformer in self._transformers.items():
            column = data.pop(column_name)
            transformed = transformer.transform(column)

            shape = transformed.shape

            if len(shape) == 2 and shape[1] == 2:
                data[column_name] = transformed[:, 0]
                new_column = '{}#{}'.format(column_name, 1)
                data[new_column] = transformed[:, 1]

            else:
                data[column_name] = transformed

        return data

    def fit_transform(self, data):
        """Prepare transformers before convert and does the required transformations to the data.

        Args:
            data (pandas.DataFrame):
                Data to transform.

        Returns:
            pandas.DataFrame
        """
        self.fit(data)
        return self.transform(data)

    @staticmethod
    def _get_columns(data, column_name):
        """Get one or more columns that match by a given name.

        Args:
            data (pandas.DataFrame):
                Table to perform the matching.
            column_name (str):
                Name to match the columns.

        Returns:
            * pandas.Series: When a single column is found.
            * pandas.DataFrame: When multiple columns are found.

        Raises:
            ValueError:
                A ``ValueError`` is raised if the match doesn't found any result.
        """
        regex = r'{}(#[0-9]+)?$'.format(column_name)
        columns = data.columns[data.columns.str.match(regex)]
        if len(columns) == 1:
            return data.pop(columns[0])

        return pd.concat([data.pop(column) for column in columns], axis=1)

    def reverse_transform(self, data):
        """Converts data back into original format.

        Not all data is reversible. When a transformer fill null values, this value
        is already in the original data and we haven't created the null column data
        can't be reversed.

        Args:
            data (pandas.DataFrame):
                Data to transform.

        Returns:
            pandas.DataFrame
        """
        if self.copy:
            data = data.copy()

        for column_name, transformer in self._transformers.items():
            columns = self._get_columns(data, column_name)
            data[column_name] = transformer.reverse_transform(columns.values)

        return data
