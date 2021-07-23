"""Base class for all the Dataset Generators."""

from abc import ABC, abstractmethod


class BaseDatasetGenerator(ABC):
    """Parent class for all the Dataset Generators."""

    TYPE = None
    SUBTYPE = None

    @staticmethod
    @abstractmethod
    def generate(num_rows):
        """This method serves as a template for dataset generators.

        Args:
            num_rows (int):
                Number of rows to generate.
        Returns:
            numpy.ndarray of size ``num_rows``
        """
        raise NotImplementedError()
