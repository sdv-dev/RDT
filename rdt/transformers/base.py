"""BaseTransformer module."""


class BaseTransformer:
    """Base class for all transformers.

    The ``BaseTransformer`` class contains methods that must be implemented
    in order to create a new transformer. The ``fit`` method is optional,
    and ``fit_transform`` method is already implemented.
    """
    INPUT_TYPE = None
    OUTPUT_TYPES = None
    DETERMINISTIC_TRANSFORM = None
    DETERMINISTIC_REVERSE = None
    COMPOSITION_IS_IDENTITY = None
    NEXT_TRANSFORMERS = None

    def get_input_type(self):
        """Return the input type supported by the transformer.

        Returns:
            string:
                Accepted input type of the transformer.
        """
        return self.INPUT_TYPE

    def get_output_types(self):
        """Return the output types supported by the transformer.

        Returns:
            dict:
                Mapping from the transformed column names to supported data types.
        """
        return self.OUTPUT_TYPES

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
        """Return the suggested next transformer to be used.

        Returns:
            string:
                Whether or not transforming and then reverse transforming returns the input data.
        """
        return "TODO" # concatenate column name to NEXT_TRANS


    def fit(self, data, columns):
        """Fit the transformer to the data.

        Args:
            data (pandas.Series or numpy.array):
                Data to transform.
            columns (list):
                List of column names from the data.
        """
        self._columns = columns
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
        """Transform the data.

        Args:
            columns_data (pandas.Series or numpy.array):
                Data to transform.

        Returns:
            numpy.array:
                Transformed data.
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

    def fit_transform(self, data):
        """Fit the transformer to the data and then transform it.

        Args:
            data (pandas.Series or numpy.array):
                Data to transform.

        Returns:
            numpy.array:
                Transformed data.
        """
        columns_data = data[self._columns]
        return self._fit_transform(columns_data)

    def _fit_transform(self, columns_data):
        """Fit the transformer to the data and then transform it.

        Args:
            columns_data (pandas.Series or numpy.array):
                Data to transform.

        Returns:
            numpy.array:
                Transformed data.
        """
        self._fit(columns_data)
        return self._transform(columns_data)

    def reverse_transform(self, data):
        """Revert the transformations to the original values.

        Args:
            data (pandas.Series or numpy.array):
                Data to transform.

        Returns:
            pandas.Series:
                Reverted data.
        """
        output_types = self.get_output_types()
        output_columns = list(output_types.keys())
        columns_data = data[output_columns]
        reversed_columns = self._reverse_transform(columns_data)
        data[self._columns] = reversed_columns

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
