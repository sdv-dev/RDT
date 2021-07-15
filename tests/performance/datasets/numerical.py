import numpy as np

from tests.performance.datasets.base import BaseDatasetGenerator


class RandomNumericalGenerator(BaseDatasetGenerator):
    """Generator that creates dataset of random integers."""

    TYPE = 'numerical'
    SUBTYPE = 'integer'

    @staticmethod
    def generate(num_rows):
        ii32 = np.iinfo(np.int32)
        return np.random.randint(ii32.min, ii32.max, num_rows)
