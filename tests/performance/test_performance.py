"""Test whether the performance of the Transformers is the expected one."""

import importlib

import numpy as np
import pandas as pd
import pytest

from rdt.transformers import BaseTransformer
from tests.datasets import BaseDatasetGenerator
from tests.performance.profiling import profile_transformer


def get_instance(obj, **kwargs):
    """Create new instance of the ``obj`` argument.

    Args:
        obj (str):
            Full name of class to import.
    """
    instance = None
    if isinstance(obj, str):
        package, name = obj.rsplit('.', 1)
        instance = getattr(importlib.import_module(package), name)(**kwargs)

    return instance


def get_fqn(obj):
    """Get the fully qualified name of the given object."""
    return f'{obj.__module__}.{obj.__name__}'


def _get_all_dataset_generators():
    """Get a list of all dataset generators."""
    generators = []

    for generator in BaseDatasetGenerator.__subclasses__():
        generators.extend(generator.__subclasses__())

    return generators


def _build_transformer_map():
    """Build a map of data type to transformer.

    Output:
        dict:
            A mapping of data type (str) to a list of transformers.
    """
    transformers = BaseTransformer.get_subclasses()
    transformers_map = {}

    for transformer in transformers:
        input_type = transformer.get_input_type()
        input_type_transformers = transformers_map.get(input_type, [])
        input_type_transformers.append(transformer)
        transformers_map[input_type] = input_type_transformers

    return transformers_map


DATASET_SIZES = [1000, 10000, 100000]
dataset_generators = _get_all_dataset_generators()
transformer_map = _build_transformer_map()


@pytest.mark.parametrize('dataset_generator', dataset_generators)
def test_performance(dataset_generator):
    """Run the performance tests for RDT.

    This test should find all relevant transformers for the given
    dataset generator, and run the ``profile_transformer``
    method, which will assert that the memory consumption
    and times are under the maximum acceptable values.

    Input:
        dataset_generator (rdt.tests.dataset.BaseDatasetGenerator)
    """
    transformers = transformer_map.get(dataset_generator.DATA_TYPE, [])

    expected = dataset_generator.get_performance_thresholds()

    for transformer in transformers:
        transformer_instance = transformer()

        for size in DATASET_SIZES:

            out = profile_transformer(
                transformer=transformer_instance,
                dataset_generator=dataset_generator,
                fit_size=size,
                transform_size=size,
            )

            assert out['Fit Time'] < size * expected['fit']['time']
            assert out['Fit Memory'] < size * expected['fit']['memory']
            assert out['Transform Time'] < size * expected['transform']['time']
            assert out['Transform Memory'] < size * expected['transform']['memory']
            assert out['Reverse Transform Time'] < size * expected['reverse_transform']['time']
            assert out['Reverse Transform Memory'] < size * expected['reverse_transform']['memory']


def _round_to_magnitude(value):
    if value == 0:
        raise ValueError("Value cannot be exactly 0.")

    for digits in range(-15, 15):
        rounded = np.round(value, digits)
        if rounded != 0:
            return rounded

    # We should never reach this line
    raise ValueError("Value is too big")


def find_transformer_boundaries(transformer, dataset_generator, fit_size,
                                transform_size, iterations=1, multiplier=5):
    """Helper function to find valid candidate boundaries for performance tests.

    The function works by:
        - Running the profiling multiple times
        - Averaging out the values for each metric
        - Multiplying the found values by the given multiplier (default=5).
        - Rounding to the found order of magnitude

    As an example, if a method took 0.012 seconds to run, the expected output
    threshold will be set to 0.1, but if it took 0.016, it will be set to 0.2.

    Args:
        transformer (Transformer):
            Transformer instance to profile.
        dataset_generator (type):
            Dataset Generator class to use.
        fit_size (int):
            Number of values to use when fitting the transformer.
        transform_size (int):
            Number of values to use when transforming and reverse transforming.
        iterations (int):
            Number of iterations to perform.
        multiplier (int):
            The value used to multiply the average results before rounding them
            up/down. Defaults to 5.

    Returns:
        pd.Series:
            Candidate values for each metric.
    """
    results = [
        profile_transformer(transformer, dataset_generator, transform_size, fit_size)
        for _ in range(iterations)
    ]
    means = pd.DataFrame(results).mean(axis=0)
    return (means * multiplier).apply(_round_to_magnitude)
