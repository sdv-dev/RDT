"""BaseTransformer module."""


class BaseTransformer:
    """Base class for all transformers.

    The ``BaseTransformer`` class contains methods that must be implemented
    in order to create a new transformer. The ``fit`` method is optional,
    and ``fit_transform`` method is already implemented.
    """

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
