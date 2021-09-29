"""Hyper transformer module."""

import warnings

from rdt.transformers import get_default_transformer, load_transformer

FIELD_ALREADY_FIT_WARNING = (
    'This field has already been fit. Only one transformer can be specified per field.'
)


class HyperTransformer:
    """HyperTransformer class.

    The ``HyperTransformer`` class contains a collection of ``transformers`` that can be
    used to transform and reverse transform one or more columns at once.

    Args:
        field_transformers (dict or None):
            Dict mapping field names to transformer to use. The keys can be a string
            representing one field name or a tuple of multiple field names. Keys can
            also specify transformers for fields derived by other transformers by
            concatenating the name of the original field to the output name of the
            transformer using ``.`` as a separator.
        field_types (dict or None):
            Dict mapping field names to their data types.
        data_type_transformers (dict or None):
            Dict mapping data types to transformers to use for that data type.
        copy (bool):
            Whether to make a copy of the input data or not. Defaults to ``True``.
        transform_output_types (list or None):
            List of acceptable data types for the output of the ``transform`` method.
            If ``None``, only ``numerical`` types will be considered acceptable.


    Example:
        Create a simple ``HyperTransformer`` instance that will decide which transformers
        to use based on the fit data ``dtypes``.

        >>> ht = HyperTransformer()

        Create a ``HyperTransformer`` passing a dict mapping fields to data types.

        >>> field_types = {
        ...     'a': 'categorical',
        ...     'b': 'numerical
        ... }
        >>> ht = HyperTransformer(field_types=field_types)

        Create a ``HyperTransformer`` passing a ``field_transformers`` dict.
        (Note: The transformers used in this example may not exist and are just used
        to illustrate the different way that a transformer can be defined for a field).

        >>> field_transformers = {
        ...     'email': EmailTransformer(),
        ...     'email.domain': EmailDomainTransformer(),
        ...     ('year', 'month', 'day'): DateTimeTransformer()
        ... }
        >>> ht = HyperTransformer(field_transformers=field_transformers)

        Create a ``HyperTransformer`` passing a dict mapping data types to transformers.
        >>> data_type_transformers = {
        ...     'categorical': LabelEncodingTransformer(),
        ...     'numerical': NumericalTransformer()
        ... }
        >>> ht = HyperTransformer(data_type_transformers=data_type_transformers)
    """

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
        self._multi_column_fields = self._create_multi_column_fields()
        self._transformers_sequence = []
        self._output_columns = []
        self._input_columns = []
        self._temp_columns = []

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

    def _get_next_transformer(self, output_field, output_type, next_transformers):
        next_transformer = None
        if output_field in self.field_transformers:
            next_transformer = self.field_transformers[output_field]

        elif output_type not in self.transform_output_types:
            if next_transformers is not None and output_field in next_transformers:
                next_transformer = next_transformers[output_field]
            else:
                next_transformer = get_default_transformer(output_type)

        return next_transformer

    def _fit_field_transformer(self, data, field, transformer):
        """Fit a transformer to its corresponding field.

        If the transformer outputs fields that aren't ML ready, then this method
        recursively fits transformers to their outputs until they are. This method
        keeps track of which fields are temporarily created by transformers as well
        as which fields will be part of the final output from ``transform``.

        Args:
            data (pandas.DataFrame):
                Data to fit the transformer to.
            field (str or tuple):
                Name of column or tuple of columns in data that will be transformed
                by the transformer.
            transformer (Transformer):
                Instance of transformer class that will fit the data.
        """
        transformer = load_transformer(transformer)
        transformer.fit(data, field)
        self._transformers_sequence.append(transformer)

        output_types = transformer.get_output_types()
        next_transformers = transformer.get_next_transformers()
        for (output_name, output_type) in output_types.items():
            output_field = self._multi_column_fields.get(output_name, output_name)
            next_transformer = self._get_next_transformer(
                output_field, output_type, next_transformers)

            if next_transformer:
                self._temp_columns.append(output_name)
                if output_name not in data:
                    data = transformer.transform(data)

                if self._field_in_data(output_field, data):
                    self._fit_field_transformer(data, output_field, next_transformer)

            else:
                self._output_columns.append(output_name)

        return data

    def fit(self, data):
        """Fit the transformers to the data.

        Args:
            data (pandas.DataFrame):
                Data to fit the transformers to.
        """
        self._input_columns = list(data.columns)
        self._update_field_types(data)
        fitted_fields = set()

        # Loop through field_transformers that are first level
        for field in self.field_transformers:
            if self._field_in_data(field, data):
                if field in fitted_fields:
                    warnings.warn(FIELD_ALREADY_FIT_WARNING)
                else:
                    data = self._fit_field_transformer(data, field, self.field_transformers[field])
                    self._add_field_to_set(field, fitted_fields)

        for (field, data_type) in self.field_types.items():
            if field in fitted_fields:
                warnings.warn(FIELD_ALREADY_FIT_WARNING)
            else:
                if data_type in self.data_type_transformers:
                    transformer = self.data_type_transformers[data_type]
                else:
                    transformer = get_default_transformer(data_type)

                data = self._fit_field_transformer(data, field, transformer)
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
