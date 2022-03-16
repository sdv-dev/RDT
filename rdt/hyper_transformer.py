"""Hyper transformer module."""

import json
import warnings
from collections import defaultdict
from copy import deepcopy

import yaml

from rdt.errors import Error, NotFittedError
from rdt.transformers import get_default_transformer, get_transformer_instance


class HyperTransformer:
    """HyperTransformer class.

    The ``HyperTransformer`` class contains a collection of ``transformers`` that can be
    used to transform and reverse transform one or more columns at once.

    Args:
        field_transformers (dict or None):
            Dict used to overwrite the transformer used for a field. If no transformer is
            specified for a field, a default transformer is selected. The keys are fields
            which can be defined as a string of the column name or a tuple of multiple column
            names. Keys can also specify transformers for fields derived by other transformers.
            This can be done by concatenating the name of the original field to the output name
            using ``.`` as a separator (eg. {field_name}.{transformer_output_name}).
        field_sdtypes (dict or None):
            Dict mapping field names to their sdtypes. If not provided, the sdtype is
            inferred using the column's Pandas ``dtype``.
        default_sdtype_transformers (dict or None):
            Dict used to overwrite the default transformer for a sdtype. The keys are
            sdtypes and the values are Transformers or Transformer instances.
        copy (bool):
            Whether to make a copy of the input data or not. Defaults to ``True``.
        transform_output_sdtypes (list or None):
            List of acceptable sdtypes for the output of the ``transform`` method.
            If ``None``, only ``numerical`` sdtypes will be considered acceptable.


    Example:
        Create a simple ``HyperTransformer`` instance that will decide which transformers
        to use based on the fit data ``dtypes``.

        >>> ht = HyperTransformer()

        Create a ``HyperTransformer`` passing a dict mapping fields to sdtypes.

        >>> field_sdtypes = {
        ...     'a': 'categorical',
        ...     'b': 'numerical'
        ... }
        >>> ht = HyperTransformer(field_sdtypes=field_sdtypes)

        Create a ``HyperTransformer`` passing a ``field_transformers`` dict.
        (Note: The transformers used in this example may not exist and are just used
        to illustrate the different way that a transformer can be defined for a field).

        >>> field_transformers = {
        ...     'email': EmailTransformer(),
        ...     'email.domain': EmailDomainTransformer(),
        ... }
        >>> ht = HyperTransformer(field_transformers=field_transformers)

        Create a ``HyperTransformer`` passing a dict mapping sdtypes to transformers.
        >>> default_sdtype_transformers = {
        ...     'categorical': LabelEncoder(),
        ...     'numerical': FloatFormatter()
        ... }
        >>> ht = HyperTransformer(default_sdtype_transformers=default_sdtype_transformers)
    """

    # pylint: disable=too-many-instance-attributes

    _DTYPES_TO_SDTYPES = {
        'i': 'integer',
        'f': 'float',
        'O': 'categorical',
        'b': 'boolean',
        'M': 'datetime',
    }
    _DEFAULT_OUTPUT_SDTYPES = [
        'numerical',
        'float',
        'integer'
    ]

    @staticmethod
    def print_tip(text):
        """Print a text with ``Tip: `` at the start of the text.

        Args:
            text (str):
                Text to print.
        """
        print(f'Tip: {text}')  # noqa: T001

    @staticmethod
    def _add_field_to_set(field, field_set):
        if isinstance(field, tuple):
            field_set.update(field)
        else:
            field_set.add(field)  # noqa -> set can't use opreator

    @staticmethod
    def _field_in_set(field, field_set):
        if isinstance(field, tuple):
            return all(column in field_set for column in field)

        return field in field_set

    @staticmethod
    def _subset(input_list, other_list, not_in=False):
        return [
            element
            for element in input_list
            if (element in other_list) ^ not_in
        ]

    def _create_multi_column_fields(self):
        multi_column_fields = {}
        for field in list(self.field_sdtypes) + list(self.field_transformers):
            if isinstance(field, tuple):
                for column in field:
                    multi_column_fields[column] = field
        return multi_column_fields

    def _validate_field_transformers(self):
        for field in self.field_transformers:
            if self._field_in_set(field, self._specified_fields):
                raise ValueError(f'Multiple transformers specified for the field {field}. '
                                 'Each field can have at most one transformer defined in '
                                 'field_transformers.')

            self._add_field_to_set(field, self._specified_fields)

    def __init__(self, copy=True, field_sdtypes=None, default_sdtype_transformers=None,
                 field_transformers=None, transform_output_sdtypes=None):
        self.copy = copy
        self.default_sdtype_transformers = default_sdtype_transformers or {}

        # ``_provided_field_sdtypes``` contains only the sdtypes specified by the user,
        # while `field_sdtypes` contains both the sdtypes specified by the user and the
        # ones learned through ``fit``/``detect_initial_config``. Same for ``field_transformers``.
        self._provided_field_sdtypes = field_sdtypes or {}
        self.field_sdtypes = self._provided_field_sdtypes.copy()
        self._provided_field_transformers = field_transformers or {}
        self.field_transformers = self._provided_field_transformers.copy()

        self._specified_fields = set()
        self._validate_field_transformers()
        self.transform_output_sdtypes = transform_output_sdtypes or self._DEFAULT_OUTPUT_SDTYPES
        self._multi_column_fields = self._create_multi_column_fields()
        self._transformers_sequence = []
        self._output_columns = []
        self._input_columns = []
        self._fitted_fields = set()
        self._fitted = False
        self._transformers_tree = defaultdict(dict)

    @staticmethod
    def _field_in_data(field, data):
        all_columns_in_data = isinstance(field, tuple) and all(col in data for col in field)
        return field in data or all_columns_in_data

    @staticmethod
    def _validate_config(config):
        sdtypes = config.get('sdtypes', {})
        transformers = config.get('transformers', {})
        for column, transformer in transformers.items():
            input_sdtype = transformer.get_input_sdtype()
            sdtype = sdtypes.get(column)
            if input_sdtype != sdtype:
                warnings.warn(f'You are assigning a {input_sdtype} transformer to a {sdtype} '
                              f"column ('{column}'). If the transformer doesn't match the "
                              'sdtype, it may lead to errors.')

    def get_config(self):
        """Get the current ``HyperTransformer`` configuration.

        Returns:
            dict:
                A dictionary containing the following two dictionaries:
                - sdtypes: A dictionary mapping column names to their ``sdtypes``.
                - transformers: A dictionary mapping column names to their transformer instances.
        """
        return {
            'sdtypes': self.field_sdtypes,
            'transformers': self.field_transformers
        }

    def set_config(self, config):
        """Set the ``HyperTransformer`` configuration.

        This method will only update the sdtypes/transformers passed. Other previously
        learned sdtypes/transformers will not be affected.

        Args:
            config (dict):
                A dictionary containing the following two dictionaries:
                - sdtypes: A dictionary mapping column names to their ``sdtypes``.
                - transformers: A dictionary mapping column names to their transformer instances.
        """
        self._validate_config(config)
        self._provided_field_sdtypes = config['sdtypes']
        self.field_sdtypes.update(config['sdtypes'])
        self._provided_field_transformers = config['transformers']
        self.field_transformers.update(config['transformers'])

    def update_field_sdtypes(self, field_sdtypes):
        """Update the ``field_sdtypes`` dict.

        Args:
            field_sdtypes (dict):
                Mapping of fields to their sdtypes. Fields can be defined as a string
                representing a column name or a tuple of multiple column names. It will
                update the existing ``field_sdtypes`` values. Calling this method will
                require ``fit`` to be run again.
        """
        self._provided_field_sdtypes.update(field_sdtypes)
        self.field_sdtypes.update(field_sdtypes)

    def get_default_sdtype_transformers(self):
        """Get the ``default_sdtype_transformer`` dict.

        Returns:
            dict:
                The ``default_sdtype_transformers`` dictionary. The keys are
                sdtypes and the values are Transformers or Transformer instances.
        """
        return self.default_sdtype_transformers

    def update_default_sdtype_transformers(self, new_sdtype_transformers):
        """Update the ``default_sdtype_transformer`` dict.

        Args:
            new_sdtype_transformers (dict):
                Dict mapping sdtypes to the default transformer class or instance to use for
                them. This dict does not need to contain an entry for every sdtype. It will be
                used to overwrite the existing defaults. Calling this method will require ``fit``
                to be run again.
        """
        self.default_sdtype_transformers.update(new_sdtype_transformers)

    def update_transformers(self, column_name_to_transformer):
        """Update any of the transformers assigned to each of the column names.

        Args:
            column_name_to_transformer(dict):
                Dict mapping column names to transformers to be used for that column.
        """
        if self._fitted:
            warnings.warn(
                "For this change to take effect, please refit your data using 'fit' "
                "or 'fit_transform'."
            )

        if len(self.field_transformers) == 0:
            raise Error(
                'Nothing to update. Use the ``detect_initial_config`` method to pre-populate '
                'all the sdtypes and transformers from your dataset.'
            )

        for column_name, transformer in column_name_to_transformer.items():
            current_sdtype = self.field_sdtypes.get(column_name)
            if current_sdtype and current_sdtype != transformer.get_input_type():
                warnings.warn(
                    f'You are assigning a {transformer.get_input_type()} transformer '
                    f'to a {current_sdtype} column ({column_name}). '
                    "If the transformer doesn't match the sdtype, it may lead to errors."
                )

            self.field_transformers[column_name] = transformer
            self._provided_field_transformers[column_name] = transformer

    def set_first_transformers_for_fields(self, field_transformers):
        """Set the first transformer to use for certain fields.

        Args:
            field_transformers (dict):
                Dict mapping fields to a transformer class name or instance. This transformer will
                be the first used on that field when the ``HyperTransformer`` calls ``transform``.
                The fields or keys can be defined as strings representing a single column name, or
                tuples of strings representing multiple column names. Calling this method will
                require ``fit`` to be run again.
        """
        self._provided_field_transformers.update(field_transformers)
        self.field_transformers.update(field_transformers)

    def get_transformer(self, field):
        """Get the transformer instance used for a field.

        Args:
            field (str or tuple):
                String representing a column name or a tuple of multiple column names.

        Returns:
            Transformer:
                Transformer instance used on the specified field during ``transform``.
        """
        if not self._fitted:
            raise NotFittedError

        return self._transformers_tree[field].get('transformer', None)

    def get_output_transformers(self, field):
        """Return dict mapping output columns of field to transformers used on them.

        Args:
            field (str or tuple):
                String representing a column name or a tuple of multiple column names.

        Returns:
            dict:
                Dictionary mapping the output names of the columns created after transforming the
                specified field, to the transformer instances used on them.
        """
        if not self._fitted:
            raise NotFittedError

        next_transformers = {}
        for output in self._transformers_tree[field].get('outputs', []):
            next_transformers[output] = self._transformers_tree[output].get('transformer', None)

        return next_transformers

    def get_final_output_columns(self, field):
        """Return list of all final output columns related to a field.

        The ``HyperTransformer`` will figure out which transformers to use on a field during
        ``transform``. If the outputs are not of an acceptable sdtype, they will also go
        through transformations. This method finds all the output columns that are of an
        acceptable final sdtype that originated from the specified field.

        Args:
            field (str or tuple):
                String representing a column name or a tuple of multiple column names.

        Returns:
            list:
                List of output column names that were created as a by-product of the specified
                field.
        """
        if not self._fitted:
            raise NotFittedError

        final_outputs = []
        outputs = self._transformers_tree[field].get('outputs', []).copy()
        while len(outputs) > 0:
            output = outputs.pop()
            if output in self._transformers_tree:
                outputs.extend(self._transformers_tree[output].get('outputs', []))
            else:
                final_outputs.append(output)

        return sorted(final_outputs, reverse=True)

    def get_transformer_tree_yaml(self):
        """Return yaml representation of transformers tree.

        After running ``fit``, a sequence of transformers is created to run each original column
        through. The sequence can be thought of as a tree, where each node is a field and the
        transformer used on it, and each neighbor is an output from that transformer. This method
        returns a YAML representation of this tree.

        Returns:
            string:
                YAML object representing the tree of transformers created during ``fit``. It has
                the following form:

                field1:
                    transformer: ExampleTransformer instance
                    outputs: [field1.out1, field1.out2]
                field1.out1:
                    transformer: FrequencyEncoder instance
                    outputs: [field1.out1.value]
                field1.out2:
                    transformer: FrequencyEncoder instance
                    outputs: [field1.out2.value]
        """
        modified_tree = deepcopy(self._transformers_tree)
        for field in modified_tree:
            class_name = modified_tree[field]['transformer'].__class__.__name__
            modified_tree[field]['transformer'] = class_name

        return yaml.safe_dump(dict(modified_tree))

    def _set_field_sdtype(self, data, field):
        clean_data = data[field].dropna()
        kind = clean_data.infer_objects().dtype.kind
        self.field_sdtypes[field] = self._DTYPES_TO_SDTYPES[kind]

    def _unfit(self):
        self.field_sdtypes = self._provided_field_sdtypes.copy()
        self.field_transformers = self._provided_field_transformers.copy()
        self._transformers_sequence = []
        self._input_columns = []
        self._output_columns = []
        self._fitted_fields.clear()
        self._fitted = False
        self._transformers_tree = defaultdict(dict)

    def _learn_config(self, data):
        """Unfit the HyperTransformer and learn the sdtypes and transformers of the data."""
        self._unfit()
        for field in data:
            if field not in self.field_sdtypes:
                self._set_field_sdtype(data, field)
            if field not in self.field_transformers:
                sdtype = self.field_sdtypes[field]
                if sdtype in self.default_sdtype_transformers:
                    self.field_transformers[field] = self.default_sdtype_transformers[sdtype]
                else:
                    self.field_transformers[field] = get_default_transformer(sdtype)

    def detect_initial_config(self, data):
        """Print the configuration of the data.

        This method detects the ``sdtype`` and transformer of each field in the data
        and then prints them as a json object.

        NOTE: This method completely resets the state of the ``HyperTransformer``.

        Args:
            data (pd.DataFrame):
                Data which will have its configuration detected.
        """
        # Reset the state of the HyperTransformer
        self.default_sdtype_transformers = {}
        self._provided_field_sdtypes = {}
        self._provided_field_transformers = {}

        # Set the sdtypes and transformers of all fields to their defaults
        self._learn_config(data)

        print('Detecting a new config from the data ... SUCCESS')  # noqa: T001
        print('Setting the new config ... SUCCESS')  # noqa: T001

        config = {
            'sdtypes': self.field_sdtypes,
            'transformers': {k: repr(v) for k, v in self.field_transformers.items()}
        }

        print('Config:')  # noqa: T001
        print(json.dumps(config, indent=4))  # noqa: T001

    def _get_next_transformer(self, output_field, output_sdtype, next_transformers):
        next_transformer = None
        if output_field in self.field_transformers:
            next_transformer = self.field_transformers[output_field]

        elif output_sdtype not in self.transform_output_sdtypes:
            if next_transformers is not None and output_field in next_transformers:
                next_transformer = next_transformers[output_field]
            else:
                next_transformer = get_default_transformer(output_sdtype)

        return next_transformer

    def _fit_field_transformer(self, data, field, transformer):
        """Fit a transformer to its corresponding field.

        This method fits a transformer to the specified field which can be a column
        name or tuple of column names. If the transformer outputs fields that aren't
        ML ready, then this method recursively fits transformers to their outputs until
        they are. This method keeps track of which fields are temporarily created by
        transformers as well as which fields will be part of the final output from ``transform``.

        Args:
            data (pandas.DataFrame):
                Data to fit the transformer to.
            field (str or tuple):
                Name of column or tuple of columns in data that will be transformed
                by the transformer.
            transformer (Transformer):
                Instance of transformer class that will fit the data.
        """
        transformer = get_transformer_instance(transformer)
        transformer.fit(data, field)
        self._add_field_to_set(field, self._fitted_fields)
        self._transformers_sequence.append(transformer)
        data = transformer.transform(data)

        output_sdtypes = transformer.get_output_sdtypes()
        next_transformers = transformer.get_next_transformers()
        self._transformers_tree[field]['transformer'] = transformer
        self._transformers_tree[field]['outputs'] = list(output_sdtypes)
        for (output_name, output_sdtype) in output_sdtypes.items():
            output_field = self._multi_column_fields.get(output_name, output_name)
            next_transformer = self._get_next_transformer(
                output_field, output_sdtype, next_transformers)

            if next_transformer:
                if self._field_in_data(output_field, data):
                    self._fit_field_transformer(data, output_field, next_transformer)

        return data

    def _validate_all_fields_fitted(self):
        non_fitted_fields = self._specified_fields.difference(self._fitted_fields)
        if non_fitted_fields:
            warnings.warn('The following fields were specified in the input arguments but not'
                          + f'found in the data: {non_fitted_fields}')

    def _sort_output_columns(self):
        """Sort ``_output_columns`` to follow the same order as the ``_input_columns``."""
        for input_column in self._input_columns:
            output_columns = self.get_final_output_columns(input_column)
            self._output_columns.extend(output_columns)

    def fit(self, data):
        """Fit the transformers to the data.

        Args:
            data (pandas.DataFrame):
                Data to fit the transformers to.
        """
        self._learn_config(data)
        self._input_columns = list(data.columns)
        for field in self._input_columns:
            data = self._fit_field_transformer(data, field, self.field_transformers[field])

        self._validate_all_fields_fitted()
        self._fitted = True
        self._sort_output_columns()

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
        if not self._fitted:
            raise NotFittedError

        unknown_columns = self._subset(data.columns, self._input_columns, not_in=True)
        if self.copy:
            data = data.copy()

        for transformer in self._transformers_sequence:
            data = transformer.transform(data, drop=False)

        transformed_columns = self._subset(self._output_columns, data.columns)
        return data.reindex(columns=unknown_columns + transformed_columns)

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
        if not self._fitted:
            raise NotFittedError

        unknown_columns = self._subset(data.columns, self._output_columns, not_in=True)
        for transformer in reversed(self._transformers_sequence):
            data = transformer.reverse_transform(data, drop=False)

        reversed_columns = self._subset(self._input_columns, data.columns)

        return data.reindex(columns=unknown_columns + reversed_columns)
