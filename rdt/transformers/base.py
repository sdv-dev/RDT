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

    def get_output_types(self):
        """Return the output types supported by the transformer.

        Returns:
            dict:
                Mapping from the transformed column names to supported data types.
        """
        if self.OUTPUT_TYPES:
            return {f'{self._column_prefix}.{k}': v for k, v in self.OUTPUT_TYPES.items()}
        
        return None

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
        if self.NEXT_TRANSFORMERS:
            return {f'{self._column_prefix}.{k}': v for k, v in self.NEXT_TRANSFORMERS.items()}

        return None

    def fit(self, data, columns):
        """Fit the transformer to the `columns` of the `data`.

        Args:
            data (pandas.Series or numpy.array):
                The entire table.
            columns (list):
                List of column names from the data to transform.
        """
        self._columns = columns
        self._column_prefix = '#'.join(columns)
        while self._column_prefix in data: # make sure the `_column_prefix` is not in the data
            self._column_prefix += '#'

        columns_data = data[columns]
        self._fit(columns_data)

    def _fit(self, columns_data):
        """Fit the transformer to the data.

        Args:
            columns_data (pandas.Series or numpy.array):
                Data to transform.
        """
        raise NotImplementedError()

    def transform(self, data):
        """Transform the `self._columns` of the `data`.

        Args:
            data (pandas.Series or numpy.array):
                The entire table.

        Returns:
            pd.DataFrame:
                The entire table, containing the transformed data.
        """
        columns_data = data[self._columns]
        transformed_data = self._transform(columns_data)
        data[transformed_data.columns] = transformed_data

        return data

    def _transform(self, columns_data):
        """Transform the data.

        Args:
            columns_data (pandas.Series or numpy.array):
                Data to transform.

        Returns:
            numpy.array:
                Transformed data.
        """
        raise NotImplementedError()

    def fit_transform(self, data, columns):
        """Fit the transformer to the `columns` of the `data` and then transform them.

        Args:
            data (pandas.Series or numpy.array):
                The entire table.
            columns (list):
                List of column names from the data to transform.

        Returns:
            pd.DataFrame:
                The entire table, containing the transformed data.
        """
        self.fit(data, columns)
        return self.transform(data)

    def reverse_transform(self, data):
        """Revert the transformations to the original values.

        Args:
            data (pandas.Series or numpy.array):
                The entire table.

        Returns:
            pandas.Series:
                The entire table, containing the reverted data.
        """
        output_columns = list(self.get_output_types().keys())
        columns_data = data[output_columns]
        data[self._columns] = self._reverse_transform(columns_data)

        columns_to_drop = set(output_columns) - set(self._columns)
        data.drop(columns_to_drop) # this breaks if we run twice

        return data

    def _reverse_transform(self, columns_data):
        """Revert the transformations to the original values.

        Args:
            columns_data (pandas.Series or numpy.array):
                Data to transform.

        Returns:
            pandas.Series:
                Reverted data.
        """
        raise NotImplementedError()
