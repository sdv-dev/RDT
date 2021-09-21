"""BaseTransformer module."""


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

    @classmethod
    def get_input_type(cls):
        """Return the input type supported by the transformer.

        Returns:
            string:
                Accepted input type of the transformer.
        """
        return cls.INPUT_TYPE
    
    @staticmethod
    def _add_prefix(dictionary, column_prefix):
        # maybe add some validation of types here
        if not dictionary:
            return None

        output = {}
        for output_columns, output_type in dictionary.items():
            output[f'{column_prefix}.{output_columns}'] = output_type

        return output

    def get_output_types(self):
        """Return the output types supported by the transformer.

        Returns:
            dict:
                Mapping from the transformed column names to supported data types.
        """
        return self._add_prefix(self._OUTPUT_TYPES, self._column_prefix)

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
        return self._add_prefix(self._NEXT_TRANSFORMERS, self._column_prefix)

    def fit(self, data, columns=None):
        """Fit the transformer to the `columns` of the `data`.

        Args:
            data (pandas.DataFrame):
                The entire table.
            columns (list or str):
                List or tuple of column names from the data to transform.
                If only one column is provided, it can be passed as a string instead.
        """
        if columns is None:
            columns = list(data.columns)

        if isinstance(columns, tuple):
            columns = list(columns)

        if isinstance(columns, list):
            if len(columns) == 1:
                columns = columns[0]

        elif not isinstance(columns, str):
            raise TypeError(f'`columns` must be either a list, tuple or a string. \
                             Instead, it was passed a {type(columns)}.')

        self._column_prefix = '#'.join(columns)
        self._output_columns = list(self.get_output_types().keys())

        # make sure none of the generated `output_columns` exists in the data
        while any(output_column in data for output_column in self._output_columns):
            self._column_prefix += '#'
            self._output_columns = list(self.get_output_types().keys())

        self._columns_to_drop = set(self._output_columns) - set(columns)
        
        self._columns = columns
        self._fit(data[columns])

    def _fit(self, columns_data):
        """Fit the transformer to the data.

        Args:
            columns_data (pandas.DataFrame):
                Data to transform.
        """
        raise NotImplementedError()

    def transform(self, data):
        """Transform the `self._columns` of the `data`.

        Args:
            data (pandas.DataFrame):
                The entire table.

        Returns:
            pd.DataFrame:
                The entire table, containing the transformed data.
        """
        if self._columns not in data.columns: # what about columns being partially in the data?
            return data

        columns_data = data[self._columns]
        transformed_data = self._transform(columns_data)
        data[self._output_columns] = transformed_data # need to make sure columns properly ordered

        return data

    def _transform(self, columns_data):
        """Transform the data.

        Args:
            columns_data (pandas.DataFrame):
                Data to transform.

        Returns:
            numpy.array:
                Transformed data.
        """
        raise NotImplementedError()

    def fit_transform(self, data, columns=None):
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
        if self._output_columns not in data:
            return data

        columns_data = data[self._output_columns]
        data[self._columns] = self._reverse_transform(columns_data)
        data.drop(self._columns_to_drop) # this line will break if you run the method twice 

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
