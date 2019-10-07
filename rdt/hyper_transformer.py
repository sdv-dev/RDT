import numpy as np
import pandas as pd

from rdt import transformers


class HyperTransformer:
    """Table transformer.

    Apply transformers on multiple columns.

        transformers = {
            '<column_name>': transformer_instance,
            '<column_name>': {
                'class': transformer_class,
                'kwargs': {
                    'subtype': 'integer'
                }
            },
            '<column_name>':
                'class': 'NumberTransformer',
                'kwargs': {
                    'subtype': 'integer'
                }
            },

        }

    Args:
        TODO
        missing (bool):
            Wheter or not to handle missing values before transforming data.
            Defaults to ``True``.
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
        if isinstance(transformer, transformers.BaseTransformer):
            return transformer

        transformer_class = transformer['class']
        if not isinstance(transformer_class, transformers.BaseTransformer):
            transformer_class = cls.get_class(transformer_class)

        transformer_kwargs = transformer.get('kwargs')
        if transformer_kwargs is None:
            transformer_kwargs = dict()

        return transformer_class(**transformer_kwargs)

    def __init__(self, column_transformers=None, copy=True, anonymize=None):
        if column_transformers:
            self.transformers = {
                column_name: self._load_transformer(transformer)
                for column_name, transformer in column_transformers.items()
            }

        self.copy = copy
        self.anonymize = anonymize or dict()

    def _analyze(self, data):
        column_transformers = dict()
        for name, column in data.items():
            if column.dtype in [np.int, np.float]:
                transformer = transformers.NumericalTransformer()
            elif column.dtype == np.object:
                anonymize = self.anonymize.get(name)
                transformer = transformers.CategoricalTransformer(anonymize=anonymize)
            elif column.dtype == np.bool:
                transformer = transformers.BooleanTransformer()
            elif np.issubdtype(column.dtype, np.datetime64):
                transformer = transformers.DatetimeTransformer()
            else:
                raise ValueError('Unsupported dtype: {}'.format(column.dtype))

            column_transformers[name] = transformer

        return column_transformers

    def fit(self, data):
        if self.transformers:
            self._transformers = self.transformers
        else:
            self._transformers = self._analyze(data)

        for column_name, transformer in self._transformers.items():
            column = data[column_name]
            transformer.fit(column)

    def transform(self, data):
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
        self.fit(data)
        return self.transform(data)

    @staticmethod
    def _get_columns(data, column_name):
        regex = r'{}(#[0-9]+)?$'.format(column_name)
        columns = data.columns[data.columns.str.match(regex)]
        if len(columns) == 1:
            return data.pop(columns[0])

        return pd.concat([data.pop(column) for column in columns], axis=1)

    def reverse_transform(self, data):
        if self.copy:
            data = data.copy()

        for column_name, transformer in self._transformers.items():
            columns = self._get_columns(data, column_name)
            data[column_name] = transformer.reverse_transform(columns.values)

        return data
