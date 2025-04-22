"""Hyper transformer module."""

import inspect
import json
import logging
import warnings
from copy import deepcopy

import pandas as pd

from rdt._utils import _validate_unique_transformer_instances
from rdt.errors import (
    ConfigNotSetError,
    InvalidConfigError,
    InvalidDataError,
    NotFittedError,
    TransformerInputError,
    TransformerProcessingError,
)
from rdt.transformers import (
    BaseMultiColumnTransformer,
    BaseTransformer,
    get_class_by_transformer_name,
    get_default_transformer,
    get_transformers_by_type,
)
from rdt.transformers.utils import flatten_column_list

LOGGER = logging.getLogger(__name__)


class Config(dict):
    """Config dict for ``HyperTransformer`` with a better representation."""

    def __repr__(self):
        """Pretty print the dictionary."""
        transformers_repr = {}
        for key, value in self['transformers'].items():
            transformed_key = str(key)
            transformers_repr[transformed_key] = repr(value)

        config = {
            'sdtypes': self['sdtypes'],
            'transformers': {str(k): repr(v) for k, v in self['transformers'].items()},
        }

        printed = json.dumps(config, indent=4)
        for transformer in self['transformers'].values():
            quoted_transformer = f'"{transformer}"'
            if quoted_transformer in printed:
                printed = printed.replace(quoted_transformer, repr(transformer))

        return printed


class HyperTransformer:
    """HyperTransformer class.

    The ``HyperTransformer`` class contains a collection of ``transformers`` that can be
    used to transform and reverse transform one or more columns at once.
    """

    # pylint: disable=too-many-instance-attributes

    _DTYPES_TO_SDTYPES = {
        'u': 'numerical',
        'i': 'numerical',
        'f': 'numerical',
        'O': 'categorical',
        'b': 'boolean',
        'M': 'datetime',
    }
    _DEFAULT_OUTPUT_SDTYPES = ['numerical', 'float', 'integer']
    _REFIT_MESSAGE = (
        "For this change to take effect, please refit your data using 'fit' or 'fit_transform'."
    )
    _DETECT_CONFIG_MESSAGE = (
        'Nothing to update. Use the `detect_initial_config` method to pre-populate all the '
        'sdtypes and transformers from your dataset.'
    )
    _NOT_FIT_MESSAGE = (
        'The HyperTransformer is not ready to use. Please fit your data first using '
        "'fit' or 'fit_transform'."
    )

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
        return [element for element in input_list if (element in other_list) ^ not_in]

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
                raise ValueError(
                    f'Multiple transformers specified for the field {field}. '
                    'Each field can have at most one transformer defined in '
                    'field_transformers.'
                )

            self._add_field_to_set(field, self._specified_fields)

    def __init__(self):
        self.field_sdtypes = {}
        self.field_transformers = {}
        self._specified_fields = set()
        self._validate_field_transformers()
        self._valid_output_sdtypes = self._DEFAULT_OUTPUT_SDTYPES
        self._multi_column_fields = {}
        self._transformers_sequence = []
        self._output_columns = []
        self._input_columns = []
        self._fitted_fields = set()
        self._fitted = False
        self._modified_config = False

    @staticmethod
    def _field_in_data(field, data):
        all_columns_in_data = isinstance(field, tuple) and all(col in data for col in field)
        return field in data or all_columns_in_data

    @staticmethod
    def _get_supported_sdtypes():
        get_transformers_by_type.cache_clear()
        return get_transformers_by_type().keys()

    def get_config(self):
        """Get the current ``HyperTransformer`` configuration.

        Returns:
            dict:
                A dictionary containing the following two dictionaries:
                - sdtypes: A dictionary mapping column names to their ``sdtypes``.
                - transformers: A dictionary mapping column names to their transformer instances.
        """
        return Config({
            'sdtypes': self.field_sdtypes,
            'transformers': self.field_transformers,
        })

    @staticmethod
    def _validate_transformers(column_name_to_transformer):
        """Validate the given transformers are valid.

        Args:
            column_name_to_transformer (dict):
                Dict mapping column names to transformers to be used for that column.

        Raises:
            Error:
                Raises an error if ``column_name_to_transformer`` contains one or more
                invalid transformers.
        """
        invalid_transformers_columns = []
        for column_name, transformer in column_name_to_transformer.items():
            if transformer and not isinstance(transformer, BaseTransformer):
                invalid_transformers_columns.append(column_name)

        if invalid_transformers_columns:
            raise InvalidConfigError(
                f'Invalid transformers for columns: {invalid_transformers_columns}. '
                'Please assign an rdt transformer instance to each column name.'
            )

        _validate_unique_transformer_instances(column_name_to_transformer)

    @staticmethod
    def _validate_sdtypes(sdtypes):
        """Validate the given sdtypes are valid.

        Args:
            sdtypes (dict):
                Dict mapping column names to sdtypes to be used for that column.

        Raises:
            Error:
                Raises an error if ``sdtypes`` contains one or more invalid sdtype.
        """
        supported_sdtypes = HyperTransformer._get_supported_sdtypes()
        unsupported_sdtypes = []
        for sdtype in sdtypes.values():
            if sdtype not in supported_sdtypes:
                unsupported_sdtypes.append(sdtype)

        if unsupported_sdtypes:
            raise InvalidConfigError(
                f'Invalid sdtypes: {unsupported_sdtypes}. If you are trying to use a '
                'premium sdtype, contact info@sdv.dev about RDT Add-Ons.'
            )

    @staticmethod
    def _validate_config(config):
        if set(config.keys()) != {'sdtypes', 'transformers'}:
            raise InvalidConfigError(
                'Error: Invalid config. Please provide 2 dictionaries '
                "named 'sdtypes' and 'transformers'."
            )

        sdtypes = config['sdtypes']
        transformers = config['transformers']

        sdtype_keys = sdtypes.keys()
        transformer_keys = flatten_column_list(transformers.keys())

        is_transformer_keys_unique = len(transformer_keys) == len(set(transformer_keys))
        if not is_transformer_keys_unique:
            raise InvalidConfigError(
                'Error: Invalid config. Please provide unique keys for the sdtypes '
                'and transformers.'
            )

        if set(sdtype_keys) != set(transformer_keys):
            raise InvalidConfigError(
                "The column names in the 'sdtypes' dictionary must match the "
                "column names in the 'transformers' dictionary."
            )

        HyperTransformer._validate_sdtypes(sdtypes)
        HyperTransformer._validate_transformers(transformers)

        mismatched_columns = []
        for column_name, transformer in transformers.items():
            if transformer is None:
                continue

            columns = column_name if isinstance(column_name, tuple) else [column_name]
            for column in columns:
                sdtype = sdtypes.get(column)
                if sdtype not in transformer.get_supported_sdtypes():
                    mismatched_columns.append(column)

        if mismatched_columns:
            raise InvalidConfigError(
                "Some transformers you've assigned are not compatible with the sdtypes. "
                f'Please change the following columns: {mismatched_columns}'
            )

    def _validate_update_columns(self, update_columns):
        unknown_columns = self._subset(
            flatten_column_list(update_columns),
            self.field_sdtypes.keys(),
            not_in=True,
        )
        if unknown_columns:
            raise InvalidConfigError(
                f'Invalid column names: {unknown_columns}. These columns do not exist in the '
                "config. Use 'set_config()' to write and set your entire config at once."
            )

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
        self.field_sdtypes.update(config['sdtypes'])
        self.field_transformers.update(config['transformers'])
        self._multi_column_fields = self._create_multi_column_fields()
        self._modified_config = True
        if self._fitted:
            warnings.warn(self._REFIT_MESSAGE)

    def _validate_update_transformers_by_sdtype(
        self, sdtype, transformer, transformer_name, transformer_parameters
    ):
        if not self.field_sdtypes:
            raise ConfigNotSetError(
                'Nothing to update. Use the `detect_initial_config` method to '
                'pre-populate all the sdtypes and transformers from your dataset.'
            )

        if transformer_name is None:
            if transformer is None:
                raise InvalidConfigError("Missing required parameter 'transformer_name'.")

            if not isinstance(transformer, BaseTransformer):
                raise InvalidConfigError(
                    'Invalid transformer. Please input an rdt transformer object.'
                )

            if sdtype not in transformer.get_supported_sdtypes():
                raise InvalidConfigError(
                    "The transformer you've assigned is incompatible with the sdtype."
                )

        else:
            if (
                transformer_name not in get_class_by_transformer_name()
                or sdtype
                not in get_class_by_transformer_name()[transformer_name].get_supported_sdtypes()
            ):
                raise InvalidConfigError(
                    f"Invalid transformer name '{transformer_name}' for the '{sdtype}' sdtype."
                )

            if transformer_parameters is not None:
                transformer = get_class_by_transformer_name()[transformer_name]
                valid = inspect.signature(transformer).parameters
                invalid_parameters = {arg for arg in transformer_parameters if arg not in valid}
                if invalid_parameters:
                    raise TransformerInputError(
                        f'Invalid parameters {tuple(sorted(invalid_parameters))} '
                        f"for the '{transformer_name}'."
                    )

    def _warn_update_transformers_by_sdtype(self, transformer, transformer_name):
        if self._fitted:
            warnings.warn(self._REFIT_MESSAGE)

        if transformer_name is not None:
            if transformer is not None:
                warnings.warn(
                    "The 'transformer' parameter will no longer be supported in future versions "
                    "of the RDT. Using the 'transformer_name' parameter instead.",
                    FutureWarning,
                )

        else:
            warnings.warn(
                "The 'transformer' parameter will no longer be supported in future versions "
                "of the RDT. Please use the 'transformer_name' and 'transformer_parameters' "
                'parameters instead.',
                FutureWarning,
            )

    def _remove_column_in_multi_column_fields(self, column):
        """Remove a column that is part of a multi-column field.

        Remove the column from the tuple and modify the ``multi_column_fields``
        as well as the ``field_transformers`` dicts accordingly.

        Args:
            column (str):
                Column name to be updated.
        """
        old_tuple = self._multi_column_fields.pop(column)
        new_tuple = tuple(item for item in old_tuple if item != column)

        if len(new_tuple) == 1:
            (new_tuple,) = new_tuple
            self._multi_column_fields.pop(new_tuple, None)
        else:
            for col in new_tuple:
                self._multi_column_fields[col] = new_tuple

        self.field_transformers[new_tuple] = self.field_transformers.pop(old_tuple)

    def _update_multi_column_transformer(self):
        """Check that multi-columns mappings are valid and update them otherwise."""
        all_fields_multi_column = set()
        for columns, transformer in self.field_transformers.items():
            if isinstance(transformer, BaseMultiColumnTransformer):
                all_fields_multi_column.add(columns)

        for field in all_fields_multi_column:
            transformer = self.field_transformers[field]

            columns_to_sdtypes = self._get_columns_to_sdtypes(field)
            try:
                transformer._validate_sdtypes(  # pylint: disable=protected-access
                    columns_to_sdtypes
                )
            except TransformerInputError:
                warnings.warn(
                    f"Transformer '{transformer.get_name()}' is incompatible with the "
                    f"multi-column field '{field}'. Assigning default transformer to the columns."
                )
                del self.field_transformers[field]
                for column, sdtype in columns_to_sdtypes.items():
                    self.field_transformers[column] = deepcopy(get_default_transformer(sdtype))

        self._multi_column_fields = self._create_multi_column_fields()

    def update_transformers_by_sdtype(
        self,
        sdtype,
        transformer=None,
        transformer_name=None,
        transformer_parameters=None,
    ):
        """Update the transformers for the specified ``sdtype``.

        Given an ``sdtype`` and a ``transformer``, change all the fields of the ``sdtype``
        to use the given transformer.

        Args:
            sdtype (str):
                Semantic data type for the transformer.
            transformer (rdt.transformers.BaseTransformer):
                Transformer class or instance to be used for the given ``sdtype``.
                Note: this parameter is deprecated, use ``transformer_name`` and
                ``transformer_parameters`` instead.
            transformer_name (str):
                A string with the class name of the transformer.
            transformer_parameters (dict):
                A dict of the kwargs of the transformer.
        """
        self._validate_update_transformers_by_sdtype(
            sdtype, transformer, transformer_name, transformer_parameters
        )
        self._warn_update_transformers_by_sdtype(transformer, transformer_name)

        transformer_instance = transformer

        if transformer_name is not None:
            if transformer_parameters is not None:
                transformer_instance = get_class_by_transformer_name()[transformer_name](
                    **transformer_parameters
                )

            else:
                transformer_instance = get_class_by_transformer_name()[transformer_name]()

        for field, field_sdtype in self.field_sdtypes.items():
            if field_sdtype == sdtype:
                self.field_transformers[field] = deepcopy(transformer_instance)
                if field in self._multi_column_fields:
                    self._remove_column_in_multi_column_fields(field)

        self._multi_column_fields = self._create_multi_column_fields()
        self._update_multi_column_transformer()
        self._modified_config = True

    def update_sdtypes(self, column_name_to_sdtype):
        """Update the ``sdtypes`` for each specified column name.

        The method may also update ``field_transformers`` to match the new sdtypes.

        Args:
            column_name_to_sdtype(dict):
                Dict mapping column names to ``sdtypes`` for that column.
        """
        if len(self.field_sdtypes) == 0:
            raise ConfigNotSetError(self._DETECT_CONFIG_MESSAGE)

        update_columns = column_name_to_sdtype.keys()
        self._validate_update_columns(update_columns)
        self._validate_sdtypes(column_name_to_sdtype)

        transformers_to_update = {}
        for column, sdtype in column_name_to_sdtype.items():
            if self.field_sdtypes.get(column) == sdtype:
                continue

            column_key = self._multi_column_fields.get(column, column)
            current_transformer = self.field_transformers.get(column_key)

            if current_transformer:
                supported_sdtypes = current_transformer.get_supported_sdtypes()
                if sdtype in supported_sdtypes:
                    continue

                warnings.warn(
                    f"Sdtype '{sdtype}' is incompatible with transformer "
                    f"'{current_transformer.get_name()}'. Assigning a new transformer to it."
                )
            if column in self._multi_column_fields:
                self._remove_column_in_multi_column_fields(column)

            transformers_to_update[column] = deepcopy(get_default_transformer(sdtype))

        self.field_sdtypes.update(column_name_to_sdtype)
        self.field_transformers.update(transformers_to_update)
        LOGGER.info(
            'The transformers for these columns may change based on the new sdtype.\n'
            "Use 'get_config()' to verify the transformers."
        )

        self._multi_column_fields = self._create_multi_column_fields()
        self._update_multi_column_transformer()
        self._modified_config = True
        if self._fitted:
            warnings.warn(self._REFIT_MESSAGE)

    def _validate_updated_transformer_unique(self, column_name, transformer):
        """Validate the transformer being updated is not reused in the config."""
        if (
            transformer in self.field_transformers.values()
            and self.field_transformers.get(column_name) != transformer
        ):
            existing_column = [
                column
                for column, existing_transformer in self.field_transformers.items()
                if transformer == existing_transformer
            ][0]
            column_name = (
                f"column ('{column_name}')"
                if isinstance(column_name, str)
                else f'columns {str(column_name)}'
            )
            existing_column = (
                f"column ('{existing_column}')"
                if isinstance(existing_column, str)
                else f'columns {str(existing_column)}'
            )
            raise InvalidConfigError(
                f'The transformer for {column_name} is already assigned to '
                f'{existing_column}. Please create different transformer objects for '
                'each assignment.'
            )

    def update_transformers(self, column_name_to_transformer):
        """Update any of the transformers assigned to each of the column names.

        Args:
            column_name_to_transformer(dict):
                Dict mapping column names to transformer instances.
        """
        if self._fitted:
            warnings.warn(self._REFIT_MESSAGE)

        if len(self.field_transformers) == 0:
            raise ConfigNotSetError(self._DETECT_CONFIG_MESSAGE)

        update_columns = column_name_to_transformer.keys()
        self._validate_update_columns(update_columns)
        self._validate_transformers(column_name_to_transformer)

        for column_name, transformer in column_name_to_transformer.items():
            if transformer is not None:
                self._validate_updated_transformer_unique(column_name, transformer)

            columns = column_name if isinstance(column_name, tuple) else (column_name,)
            for column in columns:
                if transformer is not None:
                    col_sdtype = self.field_sdtypes.get(column)
                    if col_sdtype and col_sdtype not in transformer.get_supported_sdtypes():
                        raise InvalidConfigError(
                            f"Column '{column}' is a {col_sdtype} column, which is "
                            f"incompatible with the '{transformer.get_name()}' transformer."
                        )

                if len(columns) > 1 and column in self.field_transformers:
                    del self.field_transformers[column]
                elif column in self._multi_column_fields:
                    self._remove_column_in_multi_column_fields(column)

            self.field_transformers[column_name] = transformer

        self._multi_column_fields = self._create_multi_column_fields()
        self._update_multi_column_transformer()
        self._modified_config = True

    def remove_transformers(self, column_names):
        """Remove transformers for given columns.

        This will remove the transformer for a given column name and this will not be
        transformed.

        Args:
            column_names (list):
                List of columns to remove the transformers for.
        """
        unknown_columns = []
        for column_name in column_names:
            if column_name not in self.field_sdtypes:
                unknown_columns.append(column_name)

        if unknown_columns:
            raise InvalidConfigError(
                f'Invalid column names: {unknown_columns}. These columns do not exist in the '
                "config. Use 'get_config()' to see the expected values."
            )

        for column_name in column_names:
            if column_name in self._multi_column_fields:
                self._remove_column_in_multi_column_fields(column_name)

            self.field_transformers[column_name] = None

        self._update_multi_column_transformer()
        if self._fitted:
            warnings.warn(self._REFIT_MESSAGE)

    def remove_transformers_by_sdtype(self, sdtype):
        """Remove transformers for given ``sdtype``.

        This will remove the transformers for a given ``sdtype``  and those will not be
        transformed.

        Args:
            sdtype (str):
                Semantic data type for the transformers to be removed.
        """
        if sdtype not in self._get_supported_sdtypes():
            raise InvalidConfigError(
                f"Invalid sdtype '{sdtype}'. If you are trying to use a premium sdtype, "
                'contact info@sdv.dev about RDT Add-Ons.'
            )

        for column_name, column_sdtype in self.field_sdtypes.items():
            if column_sdtype == sdtype:
                if column_name in self._multi_column_fields:
                    self._remove_column_in_multi_column_fields(column_name)

                self.field_transformers[column_name] = None

        self._update_multi_column_transformer()
        if self._fitted:
            warnings.warn(self._REFIT_MESSAGE)

    def _set_field_sdtype(self, data, field):
        clean_data = data[field].dropna()
        kind = clean_data.infer_objects().dtype.kind
        self.field_sdtypes[field] = self._DTYPES_TO_SDTYPES[kind]

    def _unfit(self):
        self._transformers_sequence = []
        self._input_columns = []
        self._output_columns = []
        self._fitted_fields.clear()
        self._fitted = False

    def _learn_config(self, data):
        """Unfit the HyperTransformer and learn the sdtypes and transformers of the data."""
        self._unfit()
        for field in data:
            if field not in self.field_sdtypes:
                self._set_field_sdtype(data, field)
            if field not in self.field_transformers:
                sdtype = self.field_sdtypes[field]
                self.field_transformers[field] = deepcopy(get_default_transformer(sdtype))

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
        self.field_sdtypes = {}
        self.field_transformers = {}

        # Set the sdtypes and transformers of all fields to their defaults
        LOGGER.info('Detecting a new config from the data ... SUCCESS')
        self._learn_config(data)

        LOGGER.info('Setting the new config ... SUCCESS')

        config = Config({
            'sdtypes': self.field_sdtypes,
            'transformers': self.field_transformers,
        })

        LOGGER.info('Config:')
        LOGGER.info(str(config))

    def _get_columns_to_sdtypes(self, field):
        """Generate the ``columns_to_sdtypes`` dict for the given field.

        Args:
            field (str, tuple[str]):
                Names of the column for the multi column trnasformer.
        """
        columns_to_sdtypes = {}
        if isinstance(field, str):
            field = (field,)

        for column in field:
            columns_to_sdtypes[column] = self.field_sdtypes[column]

        return columns_to_sdtypes

    def _fit_field_transformer(self, data, field, transformer):
        """Fit a transformer to its corresponding field.

        This method fits a transformer to the specified field which can be a column
        name or tuple of column names. If the transformer outputs fields that aren't
        ML ready, then this method recursively fits transformers to their outputs until
        they are.

        Args:
            data (pandas.DataFrame):
                Data to fit the transformer to.
            field (str or tuple):
                Name of column or tuple of columns in data that will be transformed
                by the transformer.
            transformer (Transformer):
                Instance of transformer class that will fit the data.
        """
        if transformer is None:
            self._add_field_to_set(field, self._fitted_fields)
            self._output_columns.append(field)

        else:
            if isinstance(transformer, BaseMultiColumnTransformer):
                columns_to_sdtypes = self._get_columns_to_sdtypes(field)
                transformer.fit(data, columns_to_sdtypes)
            else:
                transformer.fit(data, field)

            self._transformers_sequence.append(transformer)
            data = transformer.transform(data)

            next_transformers = transformer.get_next_transformers()
            for column_name, next_transformer in next_transformers.items():
                # If the column is part of a multi-column field, and at least one column
                # isn't present in the data, then it should not fit the next transformer
                if self._field_in_data(column_name, data):
                    data = self._fit_field_transformer(data, column_name, next_transformer)

        return data

    def _validate_all_fields_fitted(self):
        non_fitted_fields = self._specified_fields.difference(self._fitted_fields)
        if non_fitted_fields:
            warnings.warn(
                'The following fields were specified in the input arguments but not '
                f'found in the data: {non_fitted_fields}'
            )

    def _validate_config_exists(self):
        if len(self.field_sdtypes) == 0 and len(self.field_transformers) == 0:
            raise ConfigNotSetError(
                "No config detected. Set the config using 'set_config' or pre-populate "
                "it automatically from your data using 'detect_initial_config' prior to "
                'fitting your data.'
            )

    def _validate_detect_config_called(self, data):
        """Assert the ``detect_initial_config`` method is correcly called before fitting."""
        self._validate_config_exists()
        fields = list(self.field_sdtypes.keys())
        missing = any(column not in data.columns for column in fields)
        unknown_columns = self._subset(data.columns, fields, not_in=True)
        if unknown_columns or missing:
            unknown_text = f' (unknown columns: {unknown_columns})' if unknown_columns else ''
            raise InvalidDataError(
                'The data you are trying to fit has different columns than the original '
                f'detected data{unknown_text}. Column names and their '
                "sdtypes must be the same. Use the method 'get_config()' to see the expected "
                'values.'
            )

    def _validate_fitted(self):
        if not self._fitted or self._modified_config:
            raise NotFittedError(self._NOT_FIT_MESSAGE)

    def reset_randomization(self):
        """Reset the generators for the anonymized columns."""
        for transformer in self.field_transformers.values():
            if transformer:
                transformer.reset_randomization()

    def fit(self, data):
        """Fit the transformers to the data.

        Args:
            data (pandas.DataFrame):
                Data to fit the transformers to.
        """
        self._validate_detect_config_called(data)
        self._unfit()
        self._input_columns = list(data.columns)
        skipped_columns = []  # skip columns in multi column transformer already fitted
        for column in self._input_columns:
            if column in skipped_columns:
                continue

            if column in self._multi_column_fields:
                field = self._multi_column_fields[column]
                field_to_skip = [col for col in field if col != column]
                skipped_columns.extend(field_to_skip)
            else:
                field = column

            data = self._fit_field_transformer(data, field, self.field_transformers[field])

        self._validate_all_fields_fitted()
        self._fitted = True
        self._modified_config = False

        # In some cases, the 'fit' method may invoke 'transformer.transform',
        # which can advance the random seed. As a result, it can lead to inconsistent
        # values for 'instance.transform' before and after calling 'reset_randomization'.
        # To ensure consistency, we call 'reset_randomization' after fitting is done.
        self.reset_randomization()

    def _transform(self, data, prevent_subset):
        self._validate_config_exists()
        self._validate_fitted()

        unknown_columns = self._subset(data.columns, self._input_columns, not_in=True)
        if prevent_subset:
            contained = all(column in self._input_columns for column in data.columns)
            is_subset = contained and len(data.columns) < len(self._input_columns)
            if unknown_columns or is_subset:
                raise InvalidDataError(
                    'The data you are trying to transform has different columns than the original '
                    'data. Column names and their sdtypes must be the same. Use the method '
                    "'get_config()' to see the expected values."
                )

        elif unknown_columns:
            raise InvalidDataError(
                'Unexpected column names in the data you are trying to transform: '
                f"{unknown_columns}. Use 'get_config()' to see the acceptable column names."
            )

        data = data.copy()
        for transformer in self._transformers_sequence:
            data = transformer.transform(data)

        transformed_columns = self._subset(self._output_columns, data.columns)
        return data.reindex(columns=transformed_columns)

    def transform_subset(self, data):
        """Transform a subset of the fitted data's columns.

        Args:
            data (pandas.DataFrame):
                Data to transform.

        Returns:
            pandas.DataFrame:
                Transformed subset.
        """
        return self._transform(data, prevent_subset=False)

    def transform(self, data):
        """Transform the data.

        Args:
            data (pandas.DataFrame):
                Data to transform.

        Returns:
            pandas.DataFrame:
                Transformed data.
        """
        return self._transform(data, prevent_subset=True)

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

    def create_anonymized_columns(self, num_rows, column_names):
        """Create the anonymized columns for this ``HyperTransformer``.

        Generate the anonymized and text based columns that use regex expressions.

        Args:
            num_rows (int):
                Number of rows to be created. Must be an integer greater than 0.
            column_names (list):
                List of column names to be created.

        Returns:
            pandas.DataFrame:
                A data frame with the newly generated columns of the size ``num_rows``.
        """
        self._validate_fitted()

        if not isinstance(num_rows, int) or num_rows <= 0:
            raise ValueError("Parameter 'num_rows' must be an integer greater than 0.")

        unknown_columns = self._subset(column_names, self._input_columns, not_in=True)
        if unknown_columns:
            raise InvalidConfigError(
                f"Unknown column name {unknown_columns}. Use 'get_config()' to see a "
                'list of valid column names.'
            )

        columns_to_generate = set()
        for column in column_names:
            if column not in self._multi_column_fields:
                columns_to_generate.add(column)
                continue

            multi_columns = self._multi_column_fields[column]
            if any(col not in column_names for col in multi_columns):
                raise InvalidConfigError(
                    f"Column '{column}' is part of a multi-column field. You must include all "
                    'columns inside the multi-column field to generate the anonymized columns.'
                )

            columns_to_generate.add(multi_columns)

        transformers = []
        for column_name in sorted(columns_to_generate):
            transformer = self.field_transformers.get(column_name)
            if not transformer.is_generator():
                raise TransformerProcessingError(
                    f"Column '{column_name}' cannot be anonymized. All columns must be assigned "
                    "to 'AnonymizedFaker', 'RegexGenerator' or other ``generator``. Use "
                    "'get_config()' to see the current transformer assignments."
                )

            transformers.append(transformer)

        data = pd.DataFrame(index=range(num_rows))
        for transformer in transformers:
            data = transformer.reverse_transform(data)

        return data

    def _reverse_transform(self, data, prevent_subset):
        self._validate_config_exists()
        self._validate_fitted()

        unknown_columns = self._subset(data.columns, self._output_columns, not_in=True)
        if unknown_columns:
            raise InvalidDataError(
                'There are unexpected column names in the data you are trying to transform. '
                f'A reverse transform is not defined for {unknown_columns}.'
            )

        if prevent_subset:
            contained = all(column in self._output_columns for column in data.columns)
            is_subset = contained and len(data.columns) < len(self._output_columns)
            if is_subset:
                raise InvalidDataError(
                    'You must provide a transformed dataset with all the columns from the '
                    'original data.'
                )

            for transformer in reversed(self._transformers_sequence):
                data = transformer.reverse_transform(data)

        else:
            for transformer in reversed(self._transformers_sequence):
                output_columns = transformer.get_output_columns()
                if output_columns and set(output_columns).issubset(data.columns):
                    data = transformer.reverse_transform(data)

        reversed_columns = self._subset(self._input_columns, data.columns)

        return data.reindex(columns=reversed_columns)

    def reverse_transform_subset(self, data):
        """Revert the transformations for a subset of the fitted columns.

        Args:
            data (pandas.DataFrame):
                Data to revert.

        Returns:
            pandas.DataFrame:
                Reversed subset.
        """
        return self._reverse_transform(data, prevent_subset=False)

    def reverse_transform(self, data):
        """Revert the transformations back to the original values.

        Args:
            data (pandas.DataFrame):
                Data to revert.

        Returns:
            pandas.DataFrame:
                reversed data.
        """
        return self._reverse_transform(data, prevent_subset=True)
