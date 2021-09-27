"""IdentityTransformer module."""

from rdt.transformers.base import BaseTransformer


class IdentityTransformer(BaseTransformer):
    """Identity transformer that produces the same data.

    This transformer is intended for testing purposes only. The transform and reverse transform
    of this data is equal to the input.
    """

    INPUT_TYPE = None
    OUTPUT_TYPES = None

    def _fit(self, columns_data):
        """Fit the transformer to the data.

        Args:
            columns_data (pandas.Series or numpy.ndarray):
                Data to fit the transformer to.
        """
        self.INPUT_TYPE = dict(columns_data.dtypes)
        self.OUTPUT_TYPES = dict(columns_data.dtypes)

    def transform(self, columns_data):
        """Return the same input data.

        Args:
            columns_data (pandas.DataFrame or pandas.Series):
                Data to transform.

        Returns:
            numpy.ndarray:
        """
        return columns_data

    def reverse_transform(self, columns_data):
        """Return the same input data.

        Args:
            columns_data (numpy.ndarray):
                Data to revert.

        Returns:
            pandas.Series
        """
        return columns_data
