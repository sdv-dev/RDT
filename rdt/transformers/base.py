"""BaseTransformer module."""
import pandas as pd


class BaseTransformer:
    """Base class for all transformers.

    The ``BaseTransformer`` class contains methods that must be implemented
    in order to create a new transformer. The ``_fit`` method is optional,
    and ``fit_transform`` method is already implemented.
    """

    INPUT_TYPE = None
    OUTPUT_TYPES = None
    DETERMINISTIC_TRANSFORM = None
    DETERMINISTIC_REVERSE = None
    COMPOSITION_IS_IDENTITY = None
    NEXT_TRANSFORMERS = None

    _columns = None
    _column_prefix = None
    _output_columns = None

    @classmethod
    def get_input_type(cls):
        """Return the input type supported by the transformer.

        Returns:
            string:
                Accepted input type of the transformer.
        """
        return cls.INPUT_TYPE

    def _add_prefix(self, dictionary):
        if not dictionary:
            return None

        output = {}
        for output_columns, output_type in dictionary.items():
            output[f'{self._column_prefix}.{output_columns}'] = output_type

        return output

    def get_output_types(self):
        """Return the output types supported by the transformer.

        Returns:
            dict:
                Mapping from the transformed column names to supported data types.
        """
        return self._add_prefix(self.OUTPUT_TYPES)

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

    def get_next_transformers(self):
        """Return the suggested next transformer to be used for each column.

        Returns:
            dict:
                Mapping from transformed column names to the transformers to apply to each column.
        """
        return self._add_prefix(self.NEXT_TRANSFORMERS)

    @staticmethod
    def _convert_if_length_one(columns):
        """Convert columns to string if it's a list of length one."""
        if len(columns) == 1:
            columns = columns[0]

        return columns

    def fit(self, data, columns):
        """Fit the transformer to the `columns` of the `data`.

        Args:
            data (pandas.DataFrame):
                The entire table.
            columns (list):
                Column names. Must be present in the data.
        """
        # make sure columns is a list where every column is in the data
        if isinstance(columns, tuple) and columns not in data:
            columns = list(columns)
        elif not isinstance(columns, list):
            columns = [columns]

        missing = set(columns) - set(data.columns)
        if missing:
            raise KeyError(f'Columns {missing} were not present in the data.')

        self._column_prefix = '#'.join(columns)
        self._output_columns = list(self.get_output_types().keys())

        # make sure none of the generated `output_columns` exists in the data
        while any(output_column in data for output_column in self._output_columns):
            self._column_prefix += '#'
            self._output_columns = list(self.get_output_types().keys())

        self._columns = columns
        columns = self._convert_if_length_one(self._columns)
        self._fit(data[columns])

    def _fit(self, columns_data):
        """Fit the transformer to the data.

        Args:
            columns_data (pandas.DataFrame or pandas.Series):
                Data to transform.
        """
        raise NotImplementedError()

    @staticmethod
    def _convert_if_series(columns, data):
        """Convert columns to pandas.Series if it's a list of length one."""
        if isinstance(data, pd.Series):
            columns = columns[0]

        return columns

    def transform(self, data):
        """Transform the `self._columns` of the `data`.

        Args:
            data (pandas.DataFrame):
                The entire table.

        Returns:
            pd.DataFrame:
                The entire table, containing the transformed data.
        """
        # if `data` doesn't have the columns that were fitted on, don't transform
        if any(column not in data.columns for column in self._columns):
            return data

        data = data.copy()

        columns = self._convert_if_length_one(self._columns)
        columns_data = data[columns]
        transformed_data = self._transform(columns_data)

        print(transformed_data)
        output_columns = self._convert_if_series(self._output_columns, transformed_data)
        print(output_columns)
        data[output_columns] = transformed_data
        data.drop(self._columns, axis=1, inplace=True)

        return data

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

    def fit_transform(self, data, columns):
        """Fit the transformer to the `columns` of the `data` and then transform them.

        Args:
            data (pandas.DataFrame):
                The entire table.
            columns (list or tuple or str):
                List or tuple of column names from the data to transform.
                If only one column is provided, it can be passed as a string instead.
                If none are passed, fits on the entire dataset.

        Returns:
            pd.DataFrame:
                The entire table, containing the transformed data.
        """
        self.fit(data, columns)
        return self.transform(data)

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
        if any(column not in data.columns for column in self._output_columns):
            return data

        data = data.copy()

        output_columns = self._convert_if_length_one(self._output_columns)
        columns_data = data[output_columns]
        reversed_data = self._reverse_transform(columns_data)

        columns = self._convert_if_series(self._columns, reversed_data)
        data[columns] = reversed_data
        data.drop(self._output_columns, axis=1, inplace=True)

        return data

    def _reverse_transform(self, columns_data):
        """Revert the transformations to the original values.

        Args:
            columns_data (pandas.DataFrame):
                Data to transform.

        Returns:
            pandas.Series:
                Reverted data.
        """
        raise NotImplementedError()
