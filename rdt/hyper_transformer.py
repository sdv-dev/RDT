import numpy as np

from rdt.transformers import (
    BooleanTransformer, CategoricalTransformer, DatetimeTransformer, NumericalTransformer,
    load_transformers)


class HyperTransformer:
    """HyperTransformer class.

    The ``HyperTransformer`` class contains a collection of ``transformers`` that can be
    used to transform and reverse transform one or more columns at once.

    Args:
        transformers (dict or None):
            dict associating column names with transformers, which can be either passed
            directly as an instance or as a dict specification. If ``None``, a simple
            ``transformers`` dict is built automatically from the data.
        copy (bool):
            Whether to make a copy of the input data or not. Defaults to ``True``.
        anonymize (dict or None):
            Dictionary specifying the names and ``faker`` categories of the categorical
            columns that need to be anonymized. Defaults to ``None``.
        dtypes (list or None):
            List of column data types to use when building the ``transformers`` dict
            automatically. If not passed, the ``DataFrame.dtypes`` are used.

    Example:
        Create a simple ``HyperTransformer`` instance that will decide which transformers
        to use based on the fit data ``dtypes``.

        >>> ht = HyperTransformer()

        Create a ``HyperTransformer`` passing a list of dtypes.

        >>> ht = HyperTransformer(dtypes=[int, 'object', np.float64, 'datetime', 'bool'])

        Create a ``HyperTransformer`` passing a ``transformers`` dict.

        >>> transformers = {
        ...     'a': NumericalTransformer(dtype=float),
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
            - ``object`` or ``category``: a ``CategoricalTransformer`` is created.
            - ``bool``: a ``BooleanTransformer`` is created.
            - ``datetime``: a ``DatetimeTransformer`` is created.

        Any other ``dtype`` is not supported and raises a ``ValueError``.

        Args:
            data (pandas.DataFrame):
                Data used to analyze the ``pandas.DataFrame`` dtypes.

        Returns:
            dict:
                Mapping of column names and transformer instances.

        Raises:
            ValueError:
                if a ``dtype`` is not supported by the `HyperTransformer``.
        """
        transformers = dict()
        dtypes = self.dtypes or data.dtypes
        if self.dtypes:
            dtypes = self.dtypes
        else:
            dtypes = [
                data[column].dropna().infer_objects()
                for column in data.columns
            ]

        for name, dtype in zip(data.columns, dtypes):
            dtype = np.dtype(dtype)
            if dtype.kind == 'i':
                transformer = NumericalTransformer(dtype=int)
            elif dtype.kind == 'f':
                transformer = NumericalTransformer(dtype=float)
            elif dtype.kind == 'O':
                anonymize = self.anonymize.get(name)
                transformer = CategoricalTransformer(anonymize=anonymize)
            elif dtype.kind == 'b':
                transformer = BooleanTransformer()
            elif dtype.kind == 'M':
                transformer = DatetimeTransformer()
            else:
                raise ValueError('Unsupported dtype: {}'.format(dtype))

            transformers[name] = transformer

        return transformers

    def fit(self, data):
        """Fit the transformers to the data.

        Args:
            data (pandas.DataFrame):
                Data to fit the transformers to.
        """
        if self.transformers:
            self._transformers = load_transformers(self.transformers)
        else:
            self._transformers = self._analyze(data)

        for column_name, transformer in self._transformers.items():
            column = data[column_name]
            transformer.fit(column)

    def transform(self, data):
        """Transform the data.

        If ``self.copy`` is ``True`` make a copy of the input data to avoid modifying it.

        Args:
            data (pandas.DataFrame):
                Data to transform.

        Returns:
            pandas.DataFrame:
                Transformed data.
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
        """Fit the transformers to the data and then transform it.

        Args:
            data (pandas.DataFrame):
                Data to transform.

        Returns:
            pandas.DataFrame:
                Transformed data.
        """
        self.fit(data)
        return self.transform(data)

    @staticmethod
    def _get_columns(data, column_name):
        """Get one or more columns that match a given name.

        Args:
            data (pandas.DataFrame):
                Table to perform the matching.
            column_name (str):
                Name to match the columns.

        Returns:
            numpy.ndarray:
                values of the matching columns

        Raises:
            ValueError:
                if no columns match.
        """
        regex = r'{}(#[0-9]+)?$'.format(column_name)
        columns = data.columns[data.columns.str.match(regex)]
        if columns.empty:
            raise ValueError('No columns match_ {}'.format(column_name))

        values = [data.pop(column).values for column in columns]

        if len(values) == 1:
            return values[0]

        return np.column_stack(values)

    def reverse_transform(self, data):
        """Revert the transformations back to the original values.

        Args:
            data (pandas.DataFrame):
                Data to revert.

        Returns:
            pandas.DataFrame:
                reversed data.
        """
        if self.copy:
            data = data.copy()

        for column_name, transformer in self._transformers.items():
            columns = self._get_columns(data, column_name)
            data[column_name] = transformer.reverse_transform(columns)

        return data
