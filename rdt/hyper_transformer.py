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

    @classmethod
    def _load_transformer(cls, transformer):
        if isinstance(transformer, transformers.BaseTransformer):
            return transformer

        transformer_class = transformer['class']
        if not issubclass(transformer_class, transformers.BaseTransformer):
            transformer_class = getattr(transformers, transformer_class)

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
        for column_name, transformers in self.transformers.items():
            column = data[column_name]
            transformer.fit(column)

    def transform(self, data):
        if self.copy:
            data = data.copy()

        for column_name, transformer in self.transformers.items():
            column = data.pop(column_name)
            transformed = transformer.transform(column)
            num_columns = transformed.shape[1]
            if num_columns == 1:
                data[column_name] = transformed
            else:
                for index in range(num_columns):
                    new_column = '{}#{}'.format(column_name, index)
                    data[new_column] = transformed[:, index]

        return data

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    @staticmethod
    def _get_columns(column_name, data):
        columns = list()
        for column in data.columns:
            if column_name in column:
                columns.append(data.pop(column_name))

        return pd.DataFrame(columns)

    def reverse_transform(self, data):
        if self.copy:
            data = data.copy()

        for column_name, transformer in self.transformers.items():
            transformed = self._get_columns(column_name, data)
            data[column_name] = transformer.reverse_transform(transformed)

        return data
