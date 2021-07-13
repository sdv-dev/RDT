class BaseGenerator:
    """Base class for all dataset generators."""

    TYPE = None
    SUBTYPE = None

    @staticmethod
    def generate(self, num_rows):
        """This method serves as a template for dataset generators.

        Args:
            num_rows (int):
                Number of rows to generate.
        Returns:
            numpy.ndarray of size ``num_rows``
        """
        pass
