"""Hyper transformer module."""

from rdt.transformers import (
    BooleanTransformer, CategoricalTransformer, DatetimeTransformer, LabelEncodingTransformer,
    NumericalTransformer, OneHotEncodingTransformer)


class HyperTransformer:
    """HyperTransformer class.

    The ``HyperTransformer`` class contains a collection of ``transformers`` that can be
    used to transform and reverse transform one or more columns at once.

    Args:
        copy (bool):
            Whether to make a copy of the input data or not. Defaults to ``True``.
        field_types (dict or None):
            Dict mapping field names to their data types.
        data_type_transformers (dict or None):
            Dict mapping data types to transformers to use for that data type.
        field_transformers (dict or None):
            Dict mapping field names to transformers to use. The keys can be a string
            representing one field name or a tuple of multiple field names. Keys can
            also specify transformers for fields derived by other transformers by
            concatenating the name of the original field to the output name of the
            transformer using ``.`` as a separator.
        transform_output_types (list or None):
            List of acceptable data types for the output of the ``transform`` method.
            If ``None``, only ``numerical`` types will be considered acceptable.


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

    _TRANSFORMER_TEMPLATES = {
        'numerical': NumericalTransformer,
        'integer': NumericalTransformer(dtype=int),
        'float': NumericalTransformer(dtype=float),
        'categorical': CategoricalTransformer,
        'categorical_fuzzy': CategoricalTransformer(fuzzy=True),
        'one_hot_encoding': OneHotEncodingTransformer(error_on_unknown=False),
        'label_encoding': LabelEncodingTransformer,
        'boolean': BooleanTransformer,
        'datetime': DatetimeTransformer,
    }
    _DTYPE_TRANSFORMERS = {
        'i': 'numerical',
        'f': 'numerical',
        'O': 'categorical',
        'b': 'boolean',
        'M': 'datetime',
    }
    _DTYPES_TO_DATA_TYPES = {
        'i': 'integer',
        'f': 'float',
        'O': 'categorical',
        'b': 'boolean',
        'M': 'datetime',
    }
    _transformers_sequence = []
    _output_columns = []


    def __init__(self, copy=True, field_types=None, data_type_transformers=None,
                 field_transformers=None, transform_output_types=None):
        self.copy = copy
        self.field_types = field_types or dict()
        self.data_type_transformers = data_type_transformers or dict()
        self.field_transformers = field_transformers
        self.transform_output_types = transform_output_types

    def _update_field_types(self, data):
        for field in data:
            # not sure how to handle if field is part of multi-column data type
            if field not in self.field_types:
                self.field_types[field] = self._DTYPES_TO_DATA_TYPES[data[field].dtype.kind]

    def _fit_field_transformer(self, data, field, data_type):
        if field in self.field_transformers:
            transformer = self.field_transformers[field]
        elif data_type in self.data_type_transformers:
            transformer = self.data_type_transformers[data_type]
        else:
            transformer = self.DEFAULT_TRANSFORMERS[data_type]
        transformer.fit(data, field)
        self._transformers_sequence.append(transformer)
        output_types = transformer.get_output_types()
        for (output, output_type) in output_types.items():
            if output_type in self.transform_output_types:
                self._output_columns.append(output)
            else:
                transformed_data = transformer.transform(data)
                self._fit_field_transformer(transformed_data, output, output_type)

    def fit(self, data):
        """Fit the transformers to the data.

        Args:
            data (pandas.DataFrame):
                Data to fit the transformers to.
        """
        self._input_columns = data.columns
        self._update_field_types(data)

        for (field, data_type) in self.field_types.items():
            self._fit_field_transformer(data, field, data_type)

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

        for transformer in self._transformers_sequence:
            transformer.transform(data, drop=False)

        data = data.drop(self._input_columns, axis=1)
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

    def reverse_transform(self, data):
        """Revert the transformations back to the original values.

        Args:
            data (pandas.DataFrame):
                Data to revert.

        Returns:
            pandas.DataFrame:
                reversed data.
        """
        for transformer in reversed(self._transformers_sequence):
            transformer.reverse_transform(data, drop=False)

        reversed_data = data.drop(self._output_columns, axis=1)
        return reversed_data
