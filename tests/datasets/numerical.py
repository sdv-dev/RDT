"""Dataset Generators for numerical transformers."""

import numpy as np

from tests.datasets.base import BaseDatasetGenerator
from tests.datasets.utils import add_nans


class NumericalGenerator(BaseDatasetGenerator):
    """Base class for generators that create numerical data."""

    DATA_TYPE = 'numerical'

    @staticmethod
    def get_performance_thresholds():
        return {
            'fit': {
                'time': 1e-6,
                'memory': 200.0
            },
            'transform': {
                'time': 1e-6,
                'memory': 200.0
            },
            'reverse_transform': {
                'time': 2e-6,
                'memory': 500.0,
            }
        }


class RandomIntegerGenerator(NumericalGenerator):
    """Generator that creates an array of random integers."""

    @staticmethod
    def generate(num_rows):
        ii32 = np.iinfo(np.int32)
        return np.random.randint(ii32.min, ii32.max, num_rows)

    @staticmethod
    def get_performance_thresholds():
        return {
            'fit': {
                'time': 1e-05,
                'memory': 200.0
            },
            'transform': {
                'time': 2e-07,
                'memory': 10.0
            },
            'reverse_transform': {
                'time': 1e-07,
                'memory': 100.0,
            }
        }


class RandomIntegerNaNsGenerator(NumericalGenerator):
    """Generator that creates an array of random integers with nans."""

    @staticmethod
    def generate(num_rows):
        return add_nans(RandomIntegerGenerator.generate(num_rows).astype(np.float))

    @staticmethod
    def get_performance_thresholds():
        return {
            'fit': {
                'time': 1e-05,
                'memory': 100.0
            },
            'transform': {
                'time': 4e-06,
                'memory': 200.0
            },
            'reverse_transform': {
                'time': 2e-06,
                'memory': 100.0,
            }
        }


class ConstantIntegerGenerator(NumericalGenerator):
    """Generator that creates a constant array with a random integer."""

    @staticmethod
    def generate(num_rows):
        ii32 = np.iinfo(np.int32)
        constant = np.random.randint(ii32.min, ii32.max)
        return np.full(num_rows, constant)

    @staticmethod
    def get_performance_thresholds():
        return {
            'fit': {
                'time': 1e-05,
                'memory': 200.0
            },
            'transform': {
                'time': 2e-07,
                'memory': 10.0
            },
            'reverse_transform': {
                'time': 1e-07,
                'memory': 100.0,
            }
        }


class ConstantIntegerNaNsGenerator(NumericalGenerator):
    """Generator that creates a constant array with a random integer with some nans."""

    @staticmethod
    def generate(num_rows):
        return add_nans(ConstantIntegerGenerator.generate(num_rows).astype(np.float))

    @staticmethod
    def get_performance_thresholds():
        return {
            'fit': {
                'time': 1e-05,
                'memory': 100.0
            },
            'transform': {
                'time': 3e-06,
                'memory': 200.0
            },
            'reverse_transform': {
                'time': 2e-06,
                'memory': 100.0,
            }
        }


class AlmostConstantIntegerGenerator(NumericalGenerator):
    """Generator that creates an array with 2 only values, one of them repeated."""

    @staticmethod
    def generate(num_rows):
        ii32 = np.iinfo(np.int32)
        values = np.random.randint(ii32.min, ii32.max, size=2)
        additional_values = np.full(num_rows - 2, values[1])
        array = np.concatenate([values, additional_values])
        np.random.shuffle(array)
        return array

    @staticmethod
    def get_performance_thresholds():
        return {
            'fit': {
                'time': 1e-05,
                'memory': 200.0
            },
            'transform': {
                'time': 2e-07,
                'memory': 10.0
            },
            'reverse_transform': {
                'time': 1e-07,
                'memory': 100.0,
            }
        }


class AlmostConstantIntegerNaNsGenerator(NumericalGenerator):
    """Generator that creates an array with 2 only values, one of them repeated, and NaNs."""

    @staticmethod
    def generate(num_rows):
        ii32 = np.iinfo(np.int32)
        values = np.random.randint(ii32.min, ii32.max, size=2)
        additional_values = np.full(num_rows - 2, values[1]).astype(np.float)
        array = np.concatenate([values, add_nans(additional_values)])
        np.random.shuffle(array)
        return array

    @staticmethod
    def get_performance_thresholds():
        return {
            'fit': {
                'time': 1e-05,
                'memory': 100.0
            },
            'transform': {
                'time': 3e-06,
                'memory': 200.0
            },
            'reverse_transform': {
                'time': 2e-06,
                'memory': 100.0,
            }
        }


class NormalGenerator(NumericalGenerator):
    """Generator that creates an array of normally distributed float values."""

    @staticmethod
    def generate(num_rows):
        return np.random.normal(size=num_rows)

    @staticmethod
    def get_performance_thresholds():
        return {
            'fit': {
                'time': 1e-05,
                'memory': 200.0
            },
            'transform': {
                'time': 2e-07,
                'memory': 10.0
            },
            'reverse_transform': {
                'time': 1e-07,
                'memory': 100.0,
            }
        }


class NormalNaNsGenerator(NumericalGenerator):
    """Generator that creates an array of normally distributed float values, with NaNs."""

    @staticmethod
    def generate(num_rows):
        return add_nans(NormalGenerator.generate(num_rows))

    @staticmethod
    def get_performance_thresholds():
        return {
            'fit': {
                'time': 1e-05,
                'memory': 100.0
            },
            'transform': {
                'time': 4e-06,
                'memory': 200.0
            },
            'reverse_transform': {
                'time': 1e-06,
                'memory': 100.0,
            }
        }


class BigNormalGenerator(NumericalGenerator):
    """Generator that creates an array of big normally distributed float values."""

    @staticmethod
    def generate(num_rows):
        return np.random.normal(scale=1e10, size=num_rows)

    @staticmethod
    def get_performance_thresholds():
        return {
            'fit': {
                'time': 1e-05,
                'memory': 200.0
            },
            'transform': {
                'time': 2e-07,
                'memory': 10.0
            },
            'reverse_transform': {
                'time': 1e-07,
                'memory': 100.0,
            }
        }


class BigNormalNaNsGenerator(NumericalGenerator):
    """Generator that creates an array of normally distributed float values, with NaNs."""

    @staticmethod
    def generate(num_rows):
        return add_nans(BigNormalGenerator.generate(num_rows))

    @staticmethod
    def get_performance_thresholds():
        return {
            'fit': {
                'time': 1e-05,
                'memory': 100.0
            },
            'transform': {
                'time': 3e-06,
                'memory': 200.0
            },
            'reverse_transform': {
                'time': 2e-06,
                'memory': 100.0,
            }
        }
