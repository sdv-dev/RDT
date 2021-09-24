"""Dataset Generators for categorical transformers."""

import numpy as np

from tests.datasets.base import BaseDatasetGenerator
from tests.datasets.datetime import RandomGapDatetimeGenerator
from tests.datasets.utils import add_nans


class CategoricalGenerator(BaseDatasetGenerator):
    """Base class for generators that generate catgorical data."""

    DATA_TYPE = 'categorical'


class RandomIntegerGenerator(CategoricalGenerator):
    """Generator that creates an array of random integers."""

    @staticmethod
    def generate(num_rows):
        categories = [1, 2, 3, 4, 5]
        return np.random.choice(a=categories, size=num_rows)


class RandomIntegerNaNsGenerator(CategoricalGenerator):
    """Generator that creates an array of random integers with nans."""

    @staticmethod
    def generate(num_rows):
        return add_nans(RandomIntegerGenerator.generate(num_rows).astype(np.float))


class RandomStringGenerator(CategoricalGenerator):
    """Generator that creates an array of random strings."""

    @staticmethod
    def generate(num_rows):
        categories = ['Alice', 'Bob', 'Charlie', 'Dave', 'Eve']
        return np.random.choice(a=categories, size=num_rows)


class RandomStringNaNsGenerator(CategoricalGenerator):
    """Generator that creates an array of random strings with nans."""

    @staticmethod
    def generate(num_rows):
        return add_nans(RandomStringGenerator.generate(num_rows).astype('O'))


class RandomMixedGenerator(CategoricalGenerator):
    """Generator that creates an array of random mixed types.

    Mixed types include: int, float, bool, string, datetime.
    """

    @staticmethod
    def generate(num_rows):
        cat_size = 5
        categories = np.hstack([
            cat.astype('O') for cat in [
                RandomGapDatetimeGenerator.generate(cat_size),
                np.random.randint(0, 100, cat_size),
                np.random.uniform(0, 100, cat_size),
                np.arange(cat_size).astype(str),
                np.array([True, False])
            ]
        ])

        return np.random.choice(a=categories, size=num_rows)


class RandomMixedNaNsGenerator(CategoricalGenerator):
    """Generator that creates an array of random mixed types with nans.

    Mixed types include: int, float, bool, string, datetime.
    """

    @staticmethod
    def generate(num_rows):
        array = RandomMixedGenerator.generate(num_rows)

        length = len(array)
        num_nulls = np.random.randint(1, length)
        nulls_idx = np.random.choice(range(length), num_nulls)
        nulls = np.random.choice([np.nan, float('nan'), None], num_nulls)
        array[nulls_idx] = nulls

        return array


class SingleIntegerGenerator(CategoricalGenerator):
    """Generator that creates an array with a single integer."""

    @staticmethod
    def generate(num_rows):
        constant = np.random.randint(0, 100)
        return np.full(num_rows, constant)


class SingleIntegerNaNsGenerator(CategoricalGenerator):
    """Generator that creates an array with a single integer with some nans."""

    @staticmethod
    def generate(num_rows):
        return add_nans(SingleIntegerGenerator.generate(num_rows).astype(np.float))


class SingleStringGenerator(CategoricalGenerator):
    """Generator that creates an array of a single string."""

    @staticmethod
    def generate(num_rows):
        constant = 'A'
        return np.full(num_rows, constant)


class SingleStringNaNsGenerator(CategoricalGenerator):
    """Generator that creates an array of a single string with nans."""

    @staticmethod
    def generate(num_rows):
        return add_nans(SingleStringGenerator.generate(num_rows).astype('O'))


class UniqueIntegerGenerator(CategoricalGenerator):
    """Generator that creates an array of unique integers."""

    @staticmethod
    def generate(num_rows):
        return np.arange(num_rows)


class UniqueIntegerNaNsGenerator(CategoricalGenerator):
    """Generator that creates an array of unique integers with nans."""

    @staticmethod
    def generate(num_rows):
        return add_nans(UniqueIntegerGenerator.generate(num_rows))


class UniqueStringGenerator(CategoricalGenerator):
    """Generator that creates an array of unique strings."""

    @staticmethod
    def generate(num_rows):
        return np.arange(num_rows).astype(str)


class UniqueStringNaNsGenerator(CategoricalGenerator):
    """Generator that creates an array of unique strings with nans."""

    @staticmethod
    def generate(num_rows):
        return add_nans(UniqueStringGenerator.generate(num_rows).astype('O'))
