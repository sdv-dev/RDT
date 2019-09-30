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

    def __init__(self, column_transformers, copy=True):
        self.transformers = {
            column_name: self._load_transformer(transformer)
            for column_name, transformer in column_transformers.items()
        }
        self.copy = copy

    def fit(self, data):
        for column_name, transformer in self.transformers.items():
            column = data[column_name]
            transformer.fit(column)

    def transform(self, data):
        if self.copy:
            data = data.copy()

        for column_name, transformer in self.transformers.items():
            column = data.pop(column_name)
            transformed, null_column = transformer.transform(column)

            data[column_name] = transformed
            if null_column is not None:
                new_column = '{}#{}'.format(column_name, 1)
                data[new_column] = null_column

        return data

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def reverse_transform(self, data):
        if self.copy:
            data = data.copy()

        for column_name, transformer in self.transformers.items():
            data[column_name] = transformer.reverse_transform(data[column_name])

        return data
