"""IdentityTransformer module."""

from rdt.transformers.base import BaseTransformer


class IdentityTransformer(BaseTransformer):
    """Identity transformer that produces the same data.

    This transformer is intended for testing purposes only. The transform and reverse transform
    of this data is equal to the input.
    """

    def _fit(self, data):
        """Fit the transformer to the data.

        Args:
            data (pandas.Series or numpy.ndarray):
                Data to fit the transformer to.
        """
        self.OUTPUT_TYPES = {
            column: None
            for column in self.columns
        }

    def _transform(self, data):
        """Return the same input data.

        Args:
            data (pandas.Series or numpy.ndarray):
                Data to transform.

        Returns:
            pandas.DataFrame or pandas.Series
        """
        return data

    def _reverse_transform(self, data):
        """Return the same input data.

        Args:
            data (pandas.Series or numpy.ndarray):
                Data to revert.

        Returns:
            pandas.DataFrame or pandas.Series
        """
        return data
