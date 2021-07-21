import numpy as np

from tests.performance.datasets.base import BaseDatasetGenerator


def _add_nulls(array):
    length = len(array)
    num_nulls = np.random.randint(1, length)
    nulls = np.random.choice(range(length), num_nulls)
    array[nulls] = np.nan
    return array


class RandomIntegerGenerator(BaseDatasetGenerator):
    """Generator that creates an array of random integers."""

    TYPE = 'numerical'
    SUBTYPE = 'integer'

    @staticmethod
    def generate(num_rows):
        ii32 = np.iinfo(np.int32)
        return np.random.randint(ii32.min, ii32.max, num_rows)


class RandomIntegerNaNsGenerator(BaseDatasetGenerator):
    """Generator that creates an array of random integers with nulls."""

    TYPE = 'numerical'
    SUBTYPE = 'integer'

    @staticmethod
    def generate(num_rows):
        return _add_nulls(RandomIntegerGenerator.generate(num_rows).astype(np.float))


class ConstantIntegerGenerator(BaseDatasetGenerator):
    """Generator that creates a constant array with a random integer."""

    TYPE = 'numerical'
    SUBTYPE = 'integer'

    @staticmethod
    def generate(num_rows):
        ii32 = np.iinfo(np.int32)
        constant = np.random.randint(ii32.min, ii32.max)
        return np.full(num_rows, constant)


class ConstantIntegerNaNsGenerator(BaseDatasetGenerator):
    """Generator that creates a constant array with a random integer with some nulls."""

    TYPE = 'numerical'
    SUBTYPE = 'integer'

    @staticmethod
    def generate(num_rows):
        return _add_nulls(ConstantIntegerGenerator.generate(num_rows).astype(np.float))


class AlmostConstantIntegerGenerator(BaseDatasetGenerator):
    """Generator that creates an array with 2 only values, one of them repeated."""

    TYPE = 'numerical'
    SUBTYPE = 'integer'

    @staticmethod
    def generate(num_rows):
        ii32 = np.iinfo(np.int32)
        values = np.random.randint(ii32.min, ii32.max, size=2)
        additional_values = np.full(num_rows - 2, values[1])
        array = np.concatenate([values, additional_values])
        np.random.shuffle(array)
        return array


class AlmostConstantIntegerNaNsGenerator(BaseDatasetGenerator):
    """Generator that creates an array with 2 only values, one of them repeated, and NaNs."""

    TYPE = 'numerical'
    SUBTYPE = 'integer'

    @staticmethod
    def generate(num_rows):
        ii32 = np.iinfo(np.int32)
        values = np.random.randint(ii32.min, ii32.max, size=2)
        additional_values = np.full(num_rows - 2, values[1]).astype(np.float)
        array = np.concatenate([values, _add_nulls(additional_values)])
        np.random.shuffle(array)
        return array


class NormalGenerator(BaseDatasetGenerator):
    """Generator that creates an array of normally distributed float values."""

    TYPE = 'numerical'
    SUBTYPE = 'float'

    @staticmethod
    def generate(num_rows):
        return np.random.normal(size=num_rows)


class NormalNaNsGenerator(BaseDatasetGenerator):
    """Generator that creates an array of normally distributed float values, with NaNs."""

    TYPE = 'numerical'
    SUBTYPE = 'float'

    @staticmethod
    def generate(num_rows):
        return _add_nulls(NormalGenerator.generate(num_rows))


class BigNormalGenerator(BaseDatasetGenerator):
    """Generator that creates an array of big normally distributed float values."""

    TYPE = 'numerical'
    SUBTYPE = 'float'

    @staticmethod
    def generate(num_rows):
        return np.random.normal(scale=1e10, size=num_rows)


class BigNormalNaNsGenerator(BaseDatasetGenerator):
    """Generator that creates an array of normally distributed float values, with NaNs."""

    TYPE = 'numerical'
    SUBTYPE = 'float'

    @staticmethod
    def generate(num_rows):
        return _add_nulls(BigNormalGenerator.generate(num_rows))
