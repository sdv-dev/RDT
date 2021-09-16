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
    DETERMINISTIC = None
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
        obj = {}
        for output_name, value in self.OUTPUT_TYPES.items():
            obj[f'{self.column_name}#{output_name}'] = value

        return obj

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

    def is_deterministic(self):
        """Return whether transforming and then reverse transforming is deterministic.

        Note: if this process is deterministic, the output will always be the same as the input.

        Returns:
            bool:
                Whether or not transforming and then reverse transforming is deterministic.
        """
        return self.DETERMINISTIC

    def fit(self, data):
        """Fit the transformer to the data.

        Args:
            data (pandas.Series or numpy.array):
                Data to transform.
        """
        raise NotImplementedError()

    def transform(self, data):
        """Transform the data.

        Args:
            data (pandas.Series or numpy.array):
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
        self.fit(data)
        return self.transform(data)

    def reverse_transform(self, data):
        """Revert the transformations to the original values.

        Args:
            data (pandas.Series or numpy.array):
                Data to transform.

        Returns:
            pandas.Series:
                Reverted data.
        """
        raise NotImplementedError()
