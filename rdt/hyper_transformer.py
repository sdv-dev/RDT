"""Hyper transformer module."""

from rdt.transformers import (
    BooleanTransformer, CategoricalTransformer, DatetimeTransformer, LabelEncodingTransformer,
    NumericalTransformer, OneHotEncodingTransformer, get_default_transformers, load_transformer)


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
            Dict mapping field names to transformer to use. The keys can be a string
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
    _DEFAULT_OUTPUT_TYPES = [
        'numerical',
        'float',
        'integer'
    ]
    _transformers_sequence = []
    _output_columns = []
    _input_columns = []
    _temp_columns = []

    def _create_multi_column_fields(self):
        multi_column_fields = {}
        for field in list(self.field_types) + list(self.field_transformers):
            if isinstance(field, tuple):
                for column in field:
                    multi_column_fields[column] = field
        return multi_column_fields

    def __init__(self, copy=True, field_types=None, data_type_transformers=None,
                 field_transformers=None, transform_output_types=None):
        self.copy = copy
        self.field_types = field_types or {}
        self.data_type_transformers = data_type_transformers or {}
        self.field_transformers = field_transformers or {}
        self.transform_output_types = transform_output_types or self._DEFAULT_OUTPUT_TYPES
        self._default_transformers = get_default_transformers()
        self._multi_column_fields = self._create_multi_column_fields()

    @staticmethod
    def _field_in_data(field, data):
        all_columns_in_data = isinstance(field, tuple) and all(col in data for col in field)
        return field in data or all_columns_in_data

    @staticmethod
    def _add_field_to_set(field, field_set):
        if isinstance(field, tuple):
            field_set.update(field)
        else:
            field_set.add(field)

    def _update_field_types(self, data):
        # get set of provided fields including multi-column fields
        provided_fields = set()
        for field in self.field_types.keys():
            self._add_field_to_set(field, provided_fields)

        for field in data:
            if field not in provided_fields:
                self.field_types[field] = self._DTYPES_TO_DATA_TYPES[data[field].dtype.kind]

    def _fit_field_transformer(self, data, field, transformer):
        transformer = load_transformer(transformer)
        transformer.fit(data, field)
        self._transformers_sequence.append(transformer)
        output_types = transformer.get_output_types()
        next_transformers = transformer.get_next_transformers()
        for (output, output_type) in output_types.items():
            if output in self.field_transformers:
                self._temp_columns.append(output)
                transformed_data = transformer.transform(data)
                next_transformer = self.field_transformers[output]
                self._fit_field_transformer(transformed_data, output, next_transformer)
            elif (output in self._multi_column_fields and
                  self._field_in_data(self._multi_column_fields[output], data)):
                transformed_data = transformer.transform(data)
                multi_column_field = self._multi_column_fields[output]
                self._temp_columns.append(multi_column_field)
                next_transformer = self.field_transformers[multi_column_field]
                self._fit_field_transformer(transformed_data, multi_column_field, next_transformer)
            elif output_type in self.transform_output_types:
                self._output_columns.append(output)
            elif next_transformers is not None:
                self._temp_columns.append(output)
                transformed_data = transformer.transform(data)
                next_transformer = next_transformers[output]
                self._fit_field_transformer(transformed_data, output, next_transformer)
            else:
                self._temp_columns.append(output)
                transformed_data = transformer.transform(data)
                next_transformer = self._default_transformers[output_type]
                self._fit_field_transformer(transformed_data, output, next_transformer)

    def fit(self, data):
        """Fit the transformers to the data.

        Args:
            data (pandas.DataFrame):
                Data to fit the transformers to.
        """
        self._input_columns = data.columns
        self._update_field_types(data)
        fitted_fields = set()

        for field in self.field_transformers:
            if self._field_in_data(field, data):
                self._fit_field_transformer(data, field, self.field_transformers[field])
                self._add_field_to_set(field, fitted_fields)

        for (field, data_type) in self.field_types.items():
            if field not in fitted_fields:
                if data_type in self.data_type_transformers:
                    transformer = self.data_type_transformers[data_type]
                else:
                    transformer = self._default_transformers[data_type]

                self._fit_field_transformer(data, field, transformer)
                self._add_field_to_set(field, fitted_fields)

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
            data = transformer.transform(data, drop=False)

        columns_to_drop = [
            column
            for column in data
            if column in self._input_columns or column in self._temp_columns
        ]
        data = data.drop(columns_to_drop, axis=1)
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
            data = transformer.reverse_transform(data, drop=False)

        columns_to_drop = [
            col
            for col in data
            if col in self._output_columns or col in self._temp_columns
        ]
        reversed_data = data.drop(columns_to_drop, axis=1)
        return reversed_data
