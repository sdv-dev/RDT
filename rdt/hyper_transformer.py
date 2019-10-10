import numpy as np
import pandas as pd

from rdt import transformers


class HyperTransformer:
    """HyperTransformer class.

    The ``HyperTransformer`` class contains a collection of ``transformers`` that can be used
    to transform and reverse transform one or more columns at once. This class contains
    methods that can create transformers in case that those are not provided during
    instantiation.

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
        In this example we will instantiate a ``HyperTransformer`` that will transform the
        columns ``a`` and ``b`` by passing it a ``dict`` containing the name of the column
        and the instance of the transformer to be used (for the column ``a``) and a ``dict``
        with the parameters for the column ``b`` containing the class name and the keyword
        arguments for another ``NumericalTransformer``.

        >>> from rdt.transformers import NumericalTransformer
        >>> nt = NumericalTransformer(dtype=float)
        >>> column_transformers = {
        ...     'a': nt,
        ...     'b': {
        ...         'class': 'NumericalTransformer',
        ...         'kwargs': {
        ...             'dtype': int
        ...         }
        ...     }
        ... }
        >>> ht = HyperTransformer(column_transformers)
    """
    transformers = None

    @staticmethod
    def get_class(name):
        """Get transformer from its class name.

        Args:
            name (str):
                Name of the transformer.

        Returns:
            BaseTransformer
        """
        return getattr(transformers, name)

    @classmethod
    def _load_transformer(cls, transformer):
        """Load a new instance of a ``transformer``."""
        if isinstance(transformer, transformers.BaseTransformer):
            return transformer

        transformer_class = transformer['class']
        if not isinstance(transformer_class, transformers.BaseTransformer):
            transformer_class = cls.get_class(transformer_class)

        transformer_kwargs = transformer.get('kwargs')
        if transformer_kwargs is None:
            transformer_kwargs = dict()

        return transformer_class(**transformer_kwargs)

    def __init__(self, column_transformers=None, copy=True, anonymize=None, dtypes=None):
        if column_transformers:
            self.transformers = {
                column_name: self._load_transformer(transformer)
                for column_name, transformer in column_transformers.items()
            }

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
        column_transformers = dict()
        dtypes = self.dtypes or data.dtypes
        for name, dtype in zip(data.columns, dtypes):
            if np.issubdtype(dtype, np.dtype(int)):
                transformer = transformers.NumericalTransformer(dtype=int)
            elif np.issubdtype(dtype, np.dtype(float)):
                transformer = transformers.NumericalTransformer(dtype=float)
            elif dtype == np.object:
                anonymize = self.anonymize.get(name)
                transformer = transformers.CategoricalTransformer(anonymize=anonymize)
            elif np.issubdtype(dtype, np.dtype(bool)):
                transformer = transformers.BooleanTransformer()
            elif np.issubdtype(dtype, np.datetime64):
                transformer = transformers.DatetimeTransformer()
            else:
                raise ValueError('Unsupported dtype: {}'.format(dtype))

            column_transformers[name] = transformer

        return column_transformers

    def fit(self, data):
        """Prepare transformers before convert data.

        Before fit the data, check if the transformers are already defined.
        If not, the transformers will be created analyzing the data types.

        Args:
            data (pandas.DataFrame):
                Data to fit.
        """
        if self.transformers:
            self._transformers = self.transformers
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
