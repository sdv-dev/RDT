"""BaseTransformer module."""

import abc
import contextlib
import hashlib
import inspect
import warnings
from functools import wraps

import numpy as np
import pandas as pd

from rdt.errors import TransformerInputError


@contextlib.contextmanager
def set_random_states(random_states, method_name, set_model_random_state):
    """Context manager for managing the random state.

    Args:
        random_states (dict):
            Dictionary mapping each method to its current random state.
        method_name (str):
            Name of the method to set the random state for.
        set_model_random_state (function):
            Function to set the random state for the method.
    """
    original_np_state = np.random.get_state()
    random_np_state = random_states[method_name]
    np.random.set_state(random_np_state.get_state())

    try:
        yield
    finally:
        current_np_state = np.random.RandomState()
        current_np_state.set_state(np.random.get_state())
        set_model_random_state(current_np_state, method_name)

        np.random.set_state(original_np_state)


def random_state(function):
    """Set the random state before calling the function.

    Args:
        function (Callable):
            The function to wrap around.
    """

    @wraps(function)
    def wrapper(self, *args, **kwargs):
        if self.random_states is None:
            return function(self, *args, **kwargs)

        method_name = function.__name__
        with set_random_states(self.random_states, method_name, self.set_random_state):
            return function(self, *args, **kwargs)

    return wrapper


class BaseTransformer:
    """Base class for all transformers.

    The ``BaseTransformer`` class contains methods that must be implemented
    in order to create a new transformer. The ``_fit`` method is optional,
    and ``fit_transform`` method is already implemented.
    """

    INPUT_SDTYPE = None
    SUPPORTED_SDTYPES = None
    IS_GENERATOR = None
    INITIAL_FIT_STATE = np.random.RandomState(42)

    columns = None
    column_prefix = None
    output_columns = None
    random_seed = 42
    missing_value_replacement = None
    missing_value_generation = None

    def __init__(self):
        self.output_properties = {None: {'sdtype': 'float', 'next_transformer': None}}
        self.random_states = {
            'fit': self.INITIAL_FIT_STATE,
            'transform': None,
            'reverse_transform': None,
        }

    def set_random_state(self, state, method_name):
        """Set the random state for a transformer.

        Args:
            state (numpy.random.RandomState):
                The numpy random state to set.
            method_name (str):
                The method to set it for.
        """
        if method_name not in self.random_states:
            raise ValueError(
                "'method_name' must be one of 'fit', 'transform' or 'reverse_transform'."
            )

        self.random_states[method_name] = state

    def reset_randomization(self):
        """Reset the random state for ``reverse_transform``."""
        self.random_states = {
            'fit': self.INITIAL_FIT_STATE,
            'transform': np.random.RandomState(self.random_seed),
            'reverse_transform': np.random.RandomState(self.random_seed + 1),
        }

    @property
    def model_missing_values(self):
        """Return whether or not a new column is being used to model missing values."""
        warnings.warn(
            "Future versions of RDT will not support the 'model_missing_values' parameter. "
            "Please switch to using the 'missing_value_generation' parameter instead.",
            FutureWarning,
        )
        return self.missing_value_generation == 'from_column'

    def _set_missing_value_generation(self, missing_value_generation):
        if missing_value_generation not in (None, 'from_column', 'random'):
            raise TransformerInputError(
                "'missing_value_generation' must be one of the following values: "
                "None, 'from_column' or 'random'."
            )

        self.missing_value_generation = missing_value_generation

    def _set_model_missing_values(self, model_missing_values):
        warnings.warn(
            "Future versions of RDT will not support the 'model_missing_values' parameter. "
            "Please switch to using the 'missing_value_generation' parameter to select your "
            'strategy.',
            FutureWarning,
        )
        if model_missing_values is True:
            self._set_missing_value_generation('from_column')
        elif model_missing_values is False:
            self._set_missing_value_generation('random')

    @classmethod
    def get_name(cls):
        """Return transformer name.

        Returns:
            str:
                Transformer name.
        """
        return cls.__name__

    @classmethod
    def get_subclasses(cls):
        """Recursively find subclasses of this Baseline.

        Returns:
            list:
                List of all subclasses of this class.
        """
        subclasses = []
        for subclass in cls.__subclasses__():
            if abc.ABC not in subclass.__bases__:
                subclasses.append(subclass)

            subclasses += subclass.get_subclasses()

        return subclasses

    @classmethod
    def get_input_sdtype(cls):
        """Return the input sdtype supported by the transformer.

        Returns:
            string:
                Accepted input sdtype of the transformer.
        """
        warnings.warn(
            '`get_input_sdtype` is deprecated. Please use `get_supported_sdtypes` instead.',
            FutureWarning,
        )
        return cls.get_supported_sdtypes()[0]

    @classmethod
    def get_supported_sdtypes(cls):
        """Return the supported sdtypes by the transformer.

        Returns:
            list:
                Accepted input sdtypes of the transformer.
        """
        return cls.SUPPORTED_SDTYPES or [cls.INPUT_SDTYPE]

    def _get_output_to_property(self, property_):
        output = {}
        for output_column, properties in self.output_properties.items():
            # if 'sdtype' is not in the dict, ignore the column
            if property_ not in properties:
                continue
            if output_column is None:
                output[f'{self.column_prefix}'] = properties[property_]
            else:
                output[f'{self.column_prefix}.{output_column}'] = properties[property_]

        return output

    def get_output_sdtypes(self):
        """Return the output sdtypes produced by this transformer.

        Returns:
            dict:
                Mapping from the transformed column names to the produced sdtypes.
        """
        return self._get_output_to_property('sdtype')

    def get_next_transformers(self):
        """Return the suggested next transformer to be used for each column.

        Returns:
            dict:
                Mapping from transformed column names to the transformers to apply to each column.
        """
        return self._get_output_to_property('next_transformer')

    def is_generator(self):
        """Return whether this transformer generates new data or not.

        Returns:
            bool:
                Whether this transformer generates new data or not.
        """
        return bool(self.IS_GENERATOR)

    def get_input_column(self):
        """Return input column name for transformer.

        Returns:
            str:
                Input column name.
        """
        return self.columns[0]

    def get_output_columns(self):
        """Return list of column names created in ``transform``.

        Returns:
            list:
                Names of columns created during ``transform``.
        """
        return list(self._get_output_to_property('sdtype'))

    def _store_columns(self, columns, data):
        if isinstance(columns, tuple) and columns not in data:
            columns = list(columns)
        elif not isinstance(columns, list):
            columns = [columns]

        missing = set(columns) - set(data.columns)
        if missing:
            raise KeyError(f'Columns {missing} were not present in the data.')

        self.columns = columns

    @staticmethod
    def _get_columns_data(data, columns):
        if len(columns) == 1:
            columns = columns[0]

        return data[columns].copy()

    @staticmethod
    def _add_columns_to_data(data, transformed_data, transformed_names):
        """Add new columns to a ``pandas.DataFrame``.

        Args:
            - data (pd.DataFrame):
                The ``pandas.DataFrame`` to which the new columns have to be added.
            - transformed_data (pd.DataFrame, pd.Series, np.ndarray):
                The data of the new columns to be added.
            - transformed_names (list, np.ndarray):
                The names of the new columns to be added.

        Returns:
            ``pandas.DataFrame`` with the new columns added.
        """
        if transformed_names:
            if isinstance(transformed_data, (pd.Series, np.ndarray)):
                transformed_data = pd.DataFrame(transformed_data, columns=transformed_names)

            # When '#' is added to the column_prefix of a transformer
            # the columns of transformed_data and transformed_names don't match
            transformed_data.columns = transformed_names
            data = pd.concat([data, transformed_data.set_index(data.index)], axis=1)

        return data

    def _build_output_columns(self, data):
        self.column_prefix = '#'.join(self.columns)
        self.output_columns = self.get_output_columns()

        # make sure none of the generated `output_columns` exists in the data,
        # except when a column generates another with the same name
        output_columns = set(self.output_columns) - set(self.columns)
        repeated_columns = set(output_columns) & set(data.columns)
        while repeated_columns:
            warnings.warn(
                f'The output columns {repeated_columns} generated by the {self.get_name()} '
                'transformer already exist in the data (or they have already been generated '
                "by some other transformer). Appending a '#' to the column name to distinguish "
                'between them.'
            )
            self.column_prefix += '#'
            self.output_columns = self.get_output_columns()
            output_columns = set(self.output_columns) - set(self.columns)
            repeated_columns = set(output_columns) & set(data.columns)

    def __repr__(self):
        """Represent initialization of transformer as text.

        Returns:
            str:
                The name of the transformer followed by any non-default parameters.
        """
        class_name = self.__class__.get_name()
        custom_args = []
        args = inspect.getfullargspec(self.__init__)
        keys = args.args[1:]
        instanced = {
            key: getattr(self, key)
            for key in keys
            if key != 'model_missing_values' and hasattr(self, key)  # Remove after deprecation
        }

        default_values_list = args.defaults or []
        default_arg_to_default_value = {}
        if default_values_list:
            default_keys = keys[-len(default_values_list) :]
            default_arg_to_default_value = dict(zip(default_keys, default_values_list))

        if default_arg_to_default_value == instanced:
            return f'{class_name}()'

        for arg, value in instanced.items():
            if (
                arg not in default_arg_to_default_value
                or default_arg_to_default_value[arg] != value
            ):
                custom_args.append(f'{arg}={repr(value)}')

        args_string = ', '.join(custom_args)
        return f'{class_name}({args_string})'

    def _fit(self, columns_data):
        """Fit the transformer to the data.

        Args:
            columns_data (pandas.DataFrame or pandas.Series):
                Data to transform.
        """
        raise NotImplementedError()

    def _set_seed(self, data):
        hash_value = self.columns[0]
        for _, row in data.head(5).iterrows():
            hash_value += str(row[self.columns[0]])

        hash_value = int(hashlib.sha256(hash_value.encode('utf-8')).hexdigest(), 16)
        self.random_seed = hash_value % ((2**32) - 1)  # maximum value for a seed
        self.random_states = {
            'fit': self.INITIAL_FIT_STATE,
            'transform': np.random.RandomState(self.random_seed),
            'reverse_transform': np.random.RandomState(self.random_seed + 1),
        }

    @random_state
    def fit(self, data, column):
        """Fit the transformer to a ``column`` of the ``data``.

        Args:
            data (pandas.DataFrame):
                The entire table.
            column (str):
                Column name. Must be present in the data.
        """
        self._store_columns(column, data)
        self._set_seed(data)
        columns_data = self._get_columns_data(data, self.columns)
        self._fit(columns_data)
        self._build_output_columns(data)

    def _transform(self, columns_data):
        """Transform the data.

        Args:
            columns_data (pandas.DataFrame or pandas.Series):
                Data to transform.

        Returns:
            pandas.DataFrame or pandas.Series:
                Transformed data.
        """
        raise NotImplementedError()

    @random_state
    def transform(self, data):
        """Transform the `self.columns` of the `data`.

        Args:
            data (pandas.DataFrame):
                The entire table.

        Returns:
            pd.DataFrame:
                The entire table, containing the transformed data.
        """
        # if `data` doesn't have the columns that were fitted on, don't transform
        if any(column not in data.columns for column in self.columns):
            return data

        data = data.copy()
        columns_data = self._get_columns_data(data, self.columns)
        transformed_data = self._transform(columns_data)
        data = data.drop(self.columns, axis=1)
        data = self._add_columns_to_data(data, transformed_data, self.output_columns)

        return data

    def fit_transform(self, data, column):
        """Fit the transformer to a `column` of the `data` and then transform it.

        Args:
            data (pandas.DataFrame):
                The entire table.
            column (str):
                A column name.

        Returns:
            pd.DataFrame:
                The entire table, containing the transformed data.
        """
        self.fit(data, column)
        return self.transform(data)

    def _reverse_transform(self, columns_data):
        """Revert the transformations to the original values.

        Args:
            columns_data (pandas.DataFrame or pandas.Series):
                Data to revert.

        Returns:
            pandas.DataFrame or pandas.Series:
                Reverted data.
        """
        raise NotImplementedError()

    @random_state
    def reverse_transform(self, data):
        """Revert the transformations to the original values.

        Args:
            data (pandas.DataFrame):
                The entire table.

        Returns:
            pandas.DataFrame:
                The entire table, containing the reverted data.
        """
        # if `data` doesn't have the columns that were transformed, don't reverse_transform
        if any(column not in data.columns for column in self.output_columns):
            return data

        data = data.copy()
        columns_data = self._get_columns_data(data, self.output_columns)
        original_missing_values = self.missing_value_generation
        if self.missing_value_generation is not None and pd.isna(columns_data).any().any():
            warnings.warn(
                "The 'missing_value_generation' parameter is set to '"
                f"{self.missing_value_generation}' but the data already contains missing values."
                ' Missing value generation will be skipped.',
                UserWarning,
            )
            self.missing_value_generation = None

        reversed_data = self._reverse_transform(columns_data)
        self.missing_value_generation = original_missing_values
        data = data.drop(self.output_columns, axis=1)
        data = self._add_columns_to_data(data, reversed_data, self.columns)

        return data


class BaseMultiColumnTransformer(BaseTransformer):
    """Base class for all multi column transformers.

    The ``BaseMultiColumnTransformer`` class contains methods that must be implemented
    in order to create a new multi column transformer.

    Attributes:
        columns_to_sdtypes (dict):
            Dictionary mapping each column to its sdtype.
    """

    def __init__(self):
        super().__init__()
        self.columns_to_sdtypes = {}

    def get_input_column(self):
        """Override ``get_input_column`` method from ``BaseTransformer``.

        Raise an error because for multi column transformers, ``get_input_columns``
        must be used instead.
        """
        raise NotImplementedError(
            'MultiColumnTransformers does not have a single input column.'
            'Please use ``get_input_columns`` instead.'
        )

    def get_input_columns(self):
        """Return input column name for transformer.

        Returns:
            list:
                Input column names.
        """
        return self.columns

    def _get_prefix(self):
        """Return the prefix of the output columns.

        Returns:
            str:
                Prefix of the output columns.
        """
        raise NotImplementedError()

    def _get_output_to_property(self, property_):
        self.column_prefix = self._get_prefix()
        output = {}
        for output_column, properties in self.output_properties.items():
            # if 'sdtype' is not in the dict, ignore the column
            if property_ not in properties:
                continue

            if self.column_prefix is None:
                output[f'{output_column}'] = properties[property_]
            else:
                output[f'{self.column_prefix}.{output_column}'] = properties[property_]

        return output

    def _validate_columns_to_sdtypes(self, data, columns_to_sdtypes):
        """Check that all the columns in ``columns_to_sdtypes`` are present in the data."""
        missing = set(columns_to_sdtypes.keys()) - set(data.columns)
        if missing:
            missing_to_print = ', '.join(missing)
            raise ValueError(f'Columns ({missing_to_print}) are not present in the data.')

    @classmethod
    def _validate_sdtypes(cls, columns_to_sdtypes):
        raise NotImplementedError()

    def _fit(self, data):
        """Fit the transformer to the data.

        Args:
            data (pandas.DataFrame):
                Data to transform.
        """
        raise NotImplementedError()

    @random_state
    def fit(self, data, columns_to_sdtypes):
        """Fit the transformer to a ``column`` of the ``data``.

        Args:
            data (pandas.DataFrame):
                The entire table.
            columns_to_sdtypes (dict):
                Dictionary mapping each column to its sdtype.
        """
        self._validate_columns_to_sdtypes(data, columns_to_sdtypes)
        self.columns_to_sdtypes = columns_to_sdtypes
        self._store_columns(list(self.columns_to_sdtypes.keys()), data)
        self._set_seed(data)
        columns_data = self._get_columns_data(data, self.columns)
        self._fit(columns_data)
        self._build_output_columns(data)

    def fit_transform(self, data, columns_to_sdtypes):
        """Fit the transformer to a `column` of the `data` and then transform it.

        Args:
            data (pandas.DataFrame):
                The entire table.
            columns_to_sdtypes (dict):
                Dictionary mapping each column to its sdtype.

        Returns:
            pd.DataFrame:
                The entire table, containing the transformed data.
        """
        self.fit(data, columns_to_sdtypes)
        return self.transform(data)
