"""BaseTransformer module."""
import abc
import inspect
import warnings

import numpy as np
import pandas as pd


class BaseTransformer:
    """Base class for all transformers.

    The ``BaseTransformer`` class contains methods that must be implemented
    in order to create a new transformer. The ``_fit`` method is optional,
    and ``fit_transform`` method is already implemented.
    """

    INPUT_SDTYPE = None
    SUPPORTED_SDTYPES = None
    DETERMINISTIC_TRANSFORM = None
    DETERMINISTIC_REVERSE = None
    COMPOSITION_IS_IDENTITY = None
    IS_GENERATOR = None

    columns = None
    column_prefix = None
    output_columns = None

    def __init__(self):
        self.output_properties = {None: {'sdtype': 'float', 'next_transformer': None}}

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
        return cls.INPUT_SDTYPE

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

    def is_transform_deterministic(self):
        """Return whether the transform is deterministic.

        Returns:
            bool:
                Whether or not the transform is deterministic.
        """
        return self.DETERMINISTIC_TRANSFORM

    def is_reverse_deterministic(self):
        """Return whether the reverse transform is deterministic.

        Returns:
            bool:
                Whether or not the reverse transform is deterministic.
        """
        return self.DETERMINISTIC_REVERSE

    def is_composition_identity(self):
        """Return whether composition of transform and reverse transform produces the input data.

        Returns:
            bool:
                Whether or not transforming and then reverse transforming returns the input data.
        """
        return self.COMPOSITION_IS_IDENTITY

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
        if isinstance(transformed_data, (pd.Series, np.ndarray)):
            transformed_data = pd.DataFrame(transformed_data, columns=transformed_names)

        if transformed_names:
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
        defaults = args.defaults or []
        defaults = dict(zip(keys, defaults))
        instanced = {key: getattr(self, key) for key in keys}

        if defaults == instanced:
            return f'{class_name}()'

        for arg, value in instanced.items():
            if defaults[arg] != value:
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

    def fit(self, data, column):
        """Fit the transformer to a ``column`` of the ``data``.

        Args:
            data (pandas.DataFrame):
                The entire table.
            column (str):
                Column name. Must be present in the data.
        """
        self._store_columns(column, data)

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
        reversed_data = self._reverse_transform(columns_data)
        data = data.drop(self.output_columns, axis=1)
        data = self._add_columns_to_data(data, reversed_data, self.columns)

        return data
