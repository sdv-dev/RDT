class BaseTransformer(object):
    """Base class for all transformers.

    The ``BaseTransformer`` class contains methods that must be implemented
    in order to create a new transformer. The ``fit`` method is optional,
    and ``fit_transform`` method is already implemented.
    """

    def fit(self, data):
        """Prepare the transformer to convert data.

        Args:
            data (pandas.Series or numpy.array):
                Data to transform.
        """
        pass

    def transform(self, data):
        """Does the required transformations to the data.

        Args:
            data (pandas.Series or numpy.array):
                Data to transform.

        Returns:
            numpy.array
        """
        raise NotImplementedError

    def fit_transform(self, data):
        """Prepare the transformer to convert data and return the processed data.

        Args:
            data (pandas.Series or numpy.array):
                Data to transform.

        Returns:
            numpy.array
        """
        self.fit(data)
        return self.transform(data)

    def reverse_transform(self, data):
        """Converts data back into original format.

        Args:
            data (pandas.Series or numpy.array):
                Data to transform.

        Returns:
            pandas.Series
        """
        raise NotImplementedError
