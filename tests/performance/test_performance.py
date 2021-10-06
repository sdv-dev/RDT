"""Test whether the performance of the Transformers is the expected one."""

import importlib

import numpy as np
import pandas as pd
import pytest

from rdt.transformers import get_transformers_by_type
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


def _get_dataset_sizes(data_type):
    """Get a list of (fit_size, transform_size) for each dataset generator.

    Based on the data type of the dataset generator, return the list of
    sizes to run performance tests on. Each element in this list is a tuple
    of (fit_size, transform_size).

    Input:
        input_type (str):
            The type of data that the generator returns.

    Output:
        sizes (list[tuple]):
            A list of (fit_size, transform_size) configs to run tests on.
    """
    sizes = [(s, s) for s in DATASET_SIZES]

    if data_type == 'categorical':
        sizes = [(s, max(s, 1000)) for s in DATASET_SIZES if s <= 10000]

    return sizes


DATASET_SIZES = [1000, 10000, 100000]
dataset_generators = _get_all_dataset_generators()
transformer_map = get_transformers_by_type()


def _validate_metric_against_threshold(actual, expected_unit, size):
    """Validate that the observed metric is below the expected threshold.

    Input:
        actual (int or float):
            The observed value.
        expected_unit (int or float):
            The expected unit of the metric (per row).
        size (int):
            Size of the dataset.
    """
    assert actual < size * expected_unit


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

    dataset_sizes = _get_dataset_sizes(dataset_generator.DATA_TYPE)

    for transformer in transformers:
        transformer_instance = transformer()

        for sizes in dataset_sizes:
            fit_size, transform_size = sizes

            out = profile_transformer(
                transformer=transformer_instance,
                dataset_generator=dataset_generator,
                fit_size=fit_size,
                transform_size=transform_size,
            )

            _validate_metric_against_threshold(
                out['Fit Time'], expected['fit']['time'], fit_size)
            _validate_metric_against_threshold(
                out['Fit Memory'], expected['fit']['memory'], fit_size)
            _validate_metric_against_threshold(
                out['Transform Time'], expected['transform']['time'], transform_size)
            _validate_metric_against_threshold(
                out['Transform Memory'], expected['transform']['memory'], transform_size)
            _validate_metric_against_threshold(
                out['Reverse Transform Time'],
                expected['reverse_transform']['time'],
                transform_size,
            )
            _validate_metric_against_threshold(
                out['Reverse Transform Memory'],
                expected['reverse_transform']['memory'],
                transform_size,
            )


def _round_to_magnitude(value):
    if value == 0:
        raise ValueError('Value cannot be exactly 0.')

    for digits in range(-15, 15):
        rounded = np.round(value, digits)
        if rounded != 0:
            return rounded

    # We should never reach this line
    raise ValueError('Value is too big')


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
