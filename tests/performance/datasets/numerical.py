import numpy as np

from tests.performance.datasets.base import BaseGenerator


class RandomNumericalGenerator(BaseGenerator):
    """Generator that creates dataset of random integers."""

    TYPE = 'numerical'
    SUBTYPE = 'integer'

    @staticmethod
    def generate(self, num_rows):
        ii32 = np.iinfo(np.int32)
        return np.random.randint(ii32.min, ii32.max, num_rows)
