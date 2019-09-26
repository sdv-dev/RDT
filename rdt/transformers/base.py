class BaseTransformer(object):
    """Base class for all transformers."""

    type = None

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
        """Prepare the transformer to convert data and return the processed table.

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
            data (numpy.array):
                Data to transform.

        Returns:
            numpy.array
        """
        raise NotImplementedError
