"""Dataset Generators for boolean transformers."""

import numpy as np

from tests.performance.datasets.base import BaseDatasetGenerator

MAX_PERCENT_NULL = 50  # cap the percentage of null values at 50%
MIN_PERCENT = 20  # the minimum percentage of true or false is 20%


class RandomBooleanGenerator(BaseDatasetGenerator):
    """Generator that creates dataset of random booleans."""

    TYPE = 'boolean'
    SUBTYPE = 'boolean'

    @staticmethod
    def generate(num_rows):
        """Generate an array of random booleans.

        Args:
            num_rows (int):
                Number of rows of booleans to generate.

        Returns:
            numpy.ndarray of size ``num_rows`` containing random booleans.
        """
        return np.random.choice(a=[True, False], size=num_rows)


class RandomBooleanNaNsGenerator(BaseDatasetGenerator):
    """Generator that creates an array of random booleans with nulls."""

    TYPE = 'boolean'
    SUBTYPE = 'boolean'

    @staticmethod
    def generate(num_rows):
        percent_null = np.random.randint(MIN_PERCENT, MAX_PERCENT_NULL)
        percent_true = (100 - percent_null) / 2
        percent_false = 100 - percent_true - percent_null

        return np.random.choice(
            a=[True, False, None],
            size=num_rows,
            p=[percent_true / 100, percent_false / 100, percent_null / 100],
        )


class RandomSkewedBooleanGenerator(BaseDatasetGenerator):
    """Generator that creates dataset of random booleans."""

    TYPE = 'boolean'
    SUBTYPE = 'boolean'

    @staticmethod
    def generate(num_rows):
        percent_true = np.random.randint(MIN_PERCENT, 100 - MIN_PERCENT)

        return np.random.choice(
            a=[True, False],
            size=num_rows,
            p=[percent_true / 100, (100 - percent_true) / 100],
        )


class RandomSkewedBooleanNaNsGenerator(BaseDatasetGenerator):
    """Generator that creates an array of random booleans with nulls."""

    TYPE = 'boolean'
    SUBTYPE = 'boolean'

    @staticmethod
    def generate(num_rows):
        percent_null = np.random.randint(MIN_PERCENT, MAX_PERCENT_NULL)
        percent_true = np.random.randint(MIN_PERCENT, 100 - percent_null - MIN_PERCENT)
        percent_false = 100 - percent_null - percent_true

        return np.random.choice(
            a=[True, False, None],
            size=num_rows,
            p=[percent_true / 100, percent_false / 100, percent_null / 100],
        )


class ConstantBooleanGenerator(BaseDatasetGenerator):
    """Generator that creates a constant array with either True or False"""

    TYPE = 'boolean'
    SUBTYPE = 'boolean'

    @staticmethod
    def generate(num_rows):
        constant = np.random.choice([True, False])
        return np.full(num_rows, constant)


class ConstantBooleanNaNsGenerator(BaseDatasetGenerator):
    """Generator that creates a constant array with either True or False with some nulls."""

    TYPE = 'boolean'
    SUBTYPE = 'boolean'

    @staticmethod
    def generate(num_rows):
        constant = np.random.choice([True, False])
        percent_null = np.random.randint(MIN_PERCENT, MAX_PERCENT_NULL)

        return np.random.choice(
            a=[constant, None],
            size=num_rows,
            p=[(100 - percent_null) / 100, percent_null / 100],
        )
