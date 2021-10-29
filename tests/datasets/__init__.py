"""Dataset Generators to test the RDT Transformers."""

from collections import defaultdict

from tests.datasets import boolean, categorical, datetime, numerical
from tests.datasets.base import BaseDatasetGenerator

__all__ = [
    'boolean',
    'categorical',
    'datetime',
    'numerical',
    'BaseDatasetGenerator',
]


def get_dataset_generators_by_type():
    """Build a ``dict`` mapping data types to dataset generators.

    Returns:
        dict:
            Mapping of data type to a list of dataset generators that produce
            data of that data type.
    """
    dataset_generators = defaultdict(list)
    for dataset_generator in BaseDatasetGenerator.get_subclasses():
        dataset_generators[dataset_generator.DATA_TYPE].append(dataset_generator)

    return dataset_generators
