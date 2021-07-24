"""Dataset Generators for categorical transformers."""

import numpy as np

from tests.performance.datasets.base import BaseDatasetGenerator
from tests.performance.datasets.utils import add_nans


class RandomIntegerGenerator(BaseDatasetGenerator):
    """Generator that creates an array of random integers."""

    TYPE = 'numerical'
    SUBTYPE = 'integer'

    @staticmethod
    def generate(num_rows):
        categories = [1, 2, 3, 4, 5]
        return np.random.choice(a=categories, size=num_rows)


class RandomIntegerNaNsGenerator(BaseDatasetGenerator):
    """Generator that creates an array of random integers with nans."""

    TYPE = 'numerical'
    SUBTYPE = 'float'

    @staticmethod
    def generate(num_rows):
        return add_nans(RandomIntegerGenerator.generate(num_rows).astype(np.float))


class RandomCategoricalGenerator(BaseDatasetGenerator):
    """Generator that creates an array of random categories."""

    TYPE = 'categorical'

    @staticmethod
    def generate(num_rows):
        categories = ['Alice', 'Bob', 'Charlie', 'Dave', 'Eve']
        return np.random.choice(a=categories, size=num_rows)


class RandomCategoricalNaNsGenerator(BaseDatasetGenerator):
    """Generator that creates an array of random categories with nans."""

    TYPE = 'categorical'

    @staticmethod
    def generate(num_rows):
        return add_nans(RandomCategoricalGenerator.generate(num_rows).astype('O'))


class SingleIntegerGenerator(BaseDatasetGenerator):
    """Generator that creates an array with a single integer."""

    TYPE = 'numerical'
    SUBTYPE = 'integer'

    @staticmethod
    def generate(num_rows):
        constant = np.random.randint(0, 100)
        return np.full(num_rows, constant)


class SingleIntegerNaNsGenerator(BaseDatasetGenerator):
    """Generator that creates an array with a single integer with some nans."""

    TYPE = 'numerical'
    SUBTYPE = 'float'

    @staticmethod
    def generate(num_rows):
        return add_nans(SingleIntegerGenerator.generate(num_rows).astype(np.float))


class SingleCategoricalGenerator(BaseDatasetGenerator):
    """Generator that creates an array of a single category."""

    TYPE = 'categorical'

    @staticmethod
    def generate(num_rows):
        constant = 'A'
        return np.full(num_rows, constant)


class SingleCategoricalNaNsGenerator(BaseDatasetGenerator):
    """Generator that creates an array of a single category with nans."""

    TYPE = 'categorical'

    @staticmethod
    def generate(num_rows):
        return add_nans(SingleCategoricalGenerator.generate(num_rows).astype('O'))


class UniqueIntegerGenerator(BaseDatasetGenerator):
    """Generator that creates an array of unique integers."""

    TYPE = 'numerical'
    SUBTYPE = 'integer'

    @staticmethod
    def generate(num_rows):
        return np.arange(num_rows)


class UniqueIntegerNaNsGenerator(BaseDatasetGenerator):
    """Generator that creates an array of unique integers with nans."""

    TYPE = 'numerical'
    SUBTYPE = 'float'

    @staticmethod
    def generate(num_rows):
        return add_nans(UniqueIntegerGenerator.generate(num_rows))


class UniqueCategoricalGenerator(BaseDatasetGenerator):
    """Generator that creates an array of unique categories."""

    TYPE = 'categorical'

    @staticmethod
    def generate(num_rows):
        return np.arange(num_rows).astype(str)


class UniqueCategoricalNaNsGenerator(BaseDatasetGenerator):
    """Generator that creates an array of unique categories with nans."""

    TYPE = 'categorical'

    @staticmethod
    def generate(num_rows):
        return add_nans(UniqueCategoricalGenerator.generate(num_rows).astype('O'))
