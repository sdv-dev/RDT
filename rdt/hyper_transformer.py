"""Hyper transformer module."""

import json
import warnings
from collections import defaultdict
from copy import deepcopy

import yaml

from rdt.errors import Error, NotFittedError
from rdt.transformers import (
    get_default_transformer, get_transformer_instance, get_transformers_by_type)


class Config(dict):
    """Config dict for ``HyperTransformer`` with a better representation."""

    def __init__(self):
        super().__init__()
        self._provided_field_sdtypes = {}
        self._provided_field_transformers = {}
        self['field_transformers'] = {}
        self['field_sdtypes'] = {}

    @staticmethod
    def _validate_config(config):
        sdtypes = config.get('sdtypes', {})
        transformers = config.get('transformers', {})
        for column, transformer in transformers.items():
            input_sdtype = transformer.get_input_sdtype()
            sdtype = sdtypes.get(column)
            if input_sdtype != sdtype:
                warnings.warn(
                    f'You are assigning a {input_sdtype} transformer to a {sdtype} '
                    f"column ('{column}'). If the transformer doesn't match the "
                    'sdtype, it may lead to errors.'
                )

    @staticmethod
    def _get_supported_sdtypes():
        return get_transformers_by_type().keys()

    def reset(self):
        """Reset the `field_sdtypes` and `field_transformers`."""
        self['field_sdtypes'] = self._provided_field_sdtypes.copy()
        self['field_transformers'] = self._provided_field_transformers.copy()

    def update_sdtypes(self, column_name_to_sdtype):
        """Update the ``sdtypes`` for each specified column name.

        Args:
            column_name_to_sdtype(dict):
                Dict mapping column names to ``sdtypes`` for that column.
        """
        unsupported_sdtypes = []
        transformers_to_update = {}
        for column, sdtype in column_name_to_sdtype.items():
            if sdtype not in self._get_supported_sdtypes():
                unsupported_sdtypes.append(sdtype)
            elif self.field_sdtypes.get(column) != sdtype:
                current_transformer = self.field_transformers.get(column)
                if not current_transformer or current_transformer.get_input_sdtype() != sdtype:
                    transformers_to_update[column] = get_default_transformer(sdtype)

        if unsupported_sdtypes:
            raise Error(
                f'Unsupported sdtypes ({unsupported_sdtypes}). To use ``sdtypes`` with specific '
                'semantic meanings, please contact the SDV team to update to rdt_plus. Otherwise, '
                "use 'pii' to anonymize the column."
            )

        self.field_sdtypes.update(column_name_to_sdtype)
        self.field_transformers.update(transformers_to_update)
        self._provided_field_sdtypes.update(column_name_to_sdtype)

    def update_transformers_by_sdtype(self, sdtype, transformer):
        """Update the transformers for the specified ``sdtype``.

        Given an ``sdtype`` and a ``transformer``, change all the fields of the ``sdtype``
        to use the given transformer.

        Args:
            sdtype (str):
                Semantic data type for the transformer.
            transformer (rdt.transformers.BaseTransformer):
                Transformer class or instance to be used for the given ``sdtype``.
        """
        if not self['field_sdtypes']:
            raise Error(
                'Nothing to update. Use the `detect_initial_config` method to '
                'pre-populate all the sdtypes and transformers from your dataset.'
            )

        for field, field_sdtype in self['field_sdtypes'].items():
            if field_sdtype == sdtype:
                self._provided_field_transformers[field] = transformer
                self['field_transformers'][field] = transformer

    def update_transformers(self, column_name_to_transformer):
        """Update any of the transformers assigned to each of the column names.

        Args:
            column_name_to_transformer(dict):
                Dict mapping column names to transformers to be used for that column.
        """
        for column_name, transformer in column_name_to_transformer.items():
            current_sdtype = self['field_sdtypes'].get(column_name)
            if current_sdtype and current_sdtype != transformer.get_input_sdtype():
                warnings.warn(
                    f'You are assigning a {transformer.get_input_sdtype()} transformer '
                    f'to a {current_sdtype} column ({column_name}). '
                    "If the transformer doesn't match the sdtype, it may lead to errors."
                )

            self['field_transformers'][column_name] = transformer
            self._provided_field_transformers[column_name] = transformer

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
        self['field_sdtypes'].update(config['sdtypes'])
        self._provided_field_transformers = config['transformers']
        self['field_transformers'].update(config['transformers'])

    def __repr__(self):
        """Pretty print the dictionary."""
        config = {
            'sdtypes': self['field_sdtypes'],
            'transformers': {k: repr(v) for k, v in self['field_transformers'].items()}
        }
        return json.dumps(config, indent=4)


class HyperTransformer:
    """HyperTransformer class.

    The ``HyperTransformer`` class contains a collection of ``transformers`` that can be
    used to transform and reverse transform one or more columns at once.

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
        'i': 'numerical',
        'f': 'numerical',
        'O': 'categorical',
        'b': 'boolean',
        'M': 'datetime',
    }
    _DEFAULT_OUTPUT_SDTYPES = [
        'numerical',
        'float',
        'integer'
    ]
    _REFIT_MESSAGE = (
        "For this change to take effect, please refit your data using 'fit' or 'fit_transform'."
    )
    _DETECT_CONFIG_MESSAGE = (
        'Nothing to update. Use the `detect_initial_config` method to pre-populate all the '
        'sdtypes and transformers from your dataset.'
    )

    @staticmethod
    def _user_message(text, prefix=None):
        """Print a text with an optional prefix to the user.

        Args:
            text (str):
                Text to print.
            prefix (str or None):
                A prefix to add to the front of the text before printing.
        """
        message = f'{prefix}: {text}' if prefix else text
        print(message)  # noqa: T001

    @staticmethod
    def _add_field_to_set(field, field_set):
        if isinstance(field, tuple):
            field_set.update(field)
        else:
            field_set.add(field)  # noqa -> set can't use operator

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
        for field in list(self.config['field_sdtypes']) + list(self.config['field_transformers']):
            if isinstance(field, tuple):
                for column in field:
                    multi_column_fields[column] = field

        return multi_column_fields

    def _validate_field_transformers(self):
        for field in self.config['field_transformers']:
            if self._field_in_set(field, self._specified_fields):
                raise ValueError(f'Multiple transformers specified for the field {field}. '
                                 'Each field can have at most one transformer defined in '
                                 'field_transformers.')

            self._add_field_to_set(field, self._specified_fields)

    def __init__(self):
        self._default_sdtype_transformers = {}
        self.config = Config()
        self._specified_fields = set()
        self._validate_field_transformers()
        self._valid_output_sdtypes = self._DEFAULT_OUTPUT_SDTYPES
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

    def get_config(self):
        """Get the current ``HyperTransformer`` configuration.

        Returns:
            dict:
                A dictionary containing the following two dictionaries:
                - sdtypes: A dictionary mapping column names to their ``sdtypes``.
                - transformers: A dictionary mapping column names to their transformer instances.
        """
        return self.config

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
        self.config.set_config(config)

    def update_transformers_by_sdtype(self, sdtype, transformer):
        """Update the transformers for the specified ``sdtype``.

        Given an ``sdtype`` and a ``transformer``, change all the fields of the ``sdtype``
        to use the given transformer.

        Args:
            sdtype (str):
                Semantic data type for the transformer.
            transformer (rdt.transformers.BaseTransformer):
                Transformer class or instance to be used for the given ``sdtype``.
        """
        self.config.update_transformers_by_sdtype(sdtype, transformer)
        if self._fitted:
            warnings.warn(
                'For this change to take effect, please refit your data using '
                "'fit' or 'fit_transform'."
            )

    def update_sdtypes(self, column_name_to_sdtype):
        """Update the ``sdtypes`` for each specified column name.

        Args:
            column_name_to_sdtype(dict):
                Dict mapping column names to ``sdtypes`` for that column.
        """
        if len(self.config['field_sdtypes']) == 0:
            raise Error(self._DETECT_CONFIG_MESSAGE)

        self.config.update_sdtypes(column_name_to_sdtype)
        self._user_message(
            'The transformers for these columns may change based on the new sdtype.\n'
            "Use 'get_config()' to verify the transformers.", 'Info'
        )
        if self._fitted:
            warnings.warn(self._REFIT_MESSAGE)

    def update_transformers(self, column_name_to_transformer):
        """Update any of the transformers assigned to each of the column names.

        Args:
            column_name_to_transformer(dict):
                Dict mapping column names to transformers to be used for that column.
        """
        if self._fitted:
            warnings.warn(self._REFIT_MESSAGE)
        if len(self.config['field_transformers']) == 0:
            raise Error(self._DETECT_CONFIG_MESSAGE)

        self.config.update_transformers(column_name_to_transformer)

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
        self.config['field_sdtypes'][field] = self._DTYPES_TO_SDTYPES[kind]

    def _unfit(self):
        self.config.reset()
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
            if field not in self.config['field_sdtypes']:
                self._set_field_sdtype(data, field)
            if field not in self.config['field_transformers']:
                sdtype = self.config['field_sdtypes'][field]
                if sdtype in self._default_sdtype_transformers:
                    default_transformer = self._default_sdtype_transformers[sdtype]
                    self.config['field_transformers'][field] = default_transformer
                else:
                    self.config['field_transformers'][field] = get_default_transformer(sdtype)

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
        self._default_sdtype_transformers = {}
        self.config = Config()

        # Set the sdtypes and transformers of all fields to their defaults
        self._learn_config(data)

        self._user_message('Detecting a new config from the data ... SUCCESS')
        self._user_message('Setting the new config ... SUCCESS')

        self._user_message('Config:')
        self._user_message(self.config)

    def _get_next_transformer(self, output_field, output_sdtype, next_transformers):
        next_transformer = None
        if output_field in self.config['field_transformers']:
            next_transformer = self.config['field_transformers'][output_field]

        elif output_sdtype not in self._valid_output_sdtypes:
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

    def _validate_detect_config_called(self, data):
        """Assert the ``detect_initial_config`` method is correcly called before fitting."""
        if len(self.config['field_sdtypes']) == 0 and len(self.config['field_transformers']) == 0:
            raise NotFittedError(
                "No config detected. Set the config using 'set_config' or pre-populate "
                "it automatically from your data using 'detect_initial_config' prior to "
                'fitting your data.'
            )

        fields = list(self.config['field_sdtypes'].keys())
        unknown_columns = self._subset(data.columns, fields, not_in=True)
        if unknown_columns:
            raise NotFittedError(
                'The data you are trying to fit has different columns than the original '
                f'detected data (unknown columns: {unknown_columns}). Column names and their '
                "sdtypes must be the same. Use the method 'get_config()' to see the expected "
                'values.'
            )

    def fit(self, data):
        """Fit the transformers to the data.

        Args:
            data (pandas.DataFrame):
                Data to fit the transformers to.
        """
        self._validate_detect_config_called(data)
        self._learn_config(data)
        self._input_columns = list(data.columns)
        for field in self._input_columns:
            data = self._fit_field_transformer(
                data, field, self.config['field_transformers'][field])

        self._validate_all_fields_fitted()
        self._fitted = True
        self._sort_output_columns()

    def _validate_correctly_fitted(self, data):
        """Validate the data to transform has been fitted.

        This method raises errors if ``fit`` is not been called at all or if the column names
        passed to ``fit`` are not the same as the ones passed to ``transform``.
        """
        if not self._fitted:
            raise NotFittedError

        unknown_columns = self._subset(data.columns, self._input_columns, not_in=True)
        is_subset = any(column not in data.columns for column in self._input_columns)
        if unknown_columns or is_subset:
            unknown_text = f' (unknown columns: {unknown_columns})' if unknown_columns else ''
            raise NotFittedError(
                'The data you are trying to transform has different columns than the original '
                f'fitted data{unknown_text}. Column names and their sdtypes must be the same. '
                "Use the method 'get_config()' to see the expected values."
            )

    def transform(self, data):
        """Transform the data.

        Args:
            data (pandas.DataFrame):
                Data to transform.

        Returns:
            pandas.DataFrame:
                Transformed data.
        """
        self._validate_correctly_fitted(data)
        data = data.copy()
        for transformer in self._transformers_sequence:
            data = transformer.transform(data, drop=False)

        transformed_columns = self._subset(self._output_columns, data.columns)
        return data.reindex(columns=transformed_columns)

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
