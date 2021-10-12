"""Test whether the performance of the Transformers is the expected one."""

import importlib

import numpy as np
import pandas as pd
import pytest

from rdt.transformers import get_transformers_by_type
from tests.datasets import get_dataset_generators_by_type
from tests.performance.profiling import profile_transformer

TEST_NAMES = [
    'Fit Time',
    'Transform Time',
    'Reverse Transform Time',
    'Fit Memory',
    'Transform Memory',
    'Reverse Transform Memory',
]

DATASET_SIZES = [1000, 10000, 100000]


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


def _get_dataset_sizes(data_type):
    """Get a list of (fit_size, transform_size) for each dataset generator.

    Based on the data type of the dataset generator, return the list of
    sizes to run performance tests on. Each element in this list is a tuple
    of (fit_size, transform_size).

    Args:
        input_type (str):
            The type of data that the generator returns.

    Returns:
        sizes (list[tuple]):
            A list of (fit_size, transform_size) configs to run tests on.
    """
    sizes = [(s, s) for s in DATASET_SIZES]

    if data_type == 'categorical':
        sizes = [(s, max(s, 1000)) for s in DATASET_SIZES if s <= 10000]

    return sizes


def _get_performance_test_cases():
    """Get all the (transformer, dataset_generator) combinations for testing."""
    all_test_cases = []

    dataset_generators = get_dataset_generators_by_type()
    transformers = get_transformers_by_type()

    for data_type, transformers_for_type in transformers.items():
        dataset_generators_for_type = dataset_generators.get(data_type, [])

        for transformer in transformers_for_type:
            for dataset_generator in dataset_generators_for_type:
                all_test_cases.append((transformer, dataset_generator))

    return all_test_cases


test_cases = _get_performance_test_cases()


def _validate_metric_against_threshold(actual, expected_unit, size):
    """Validate that the observed metric is below the expected threshold.

    Args:
        actual (int or float):
            The observed value.
        expected_unit (int or float):
            The expected unit of the metric (per row).
        size (int):
            Size of the dataset.
    """
    assert actual < size * expected_unit


def validate_performance(transformer, dataset_generator, should_assert=False, results=None):
    """Validate the performance of all transformers for a dataset_generator.

    Args:
        transformer (rdt.transformers.BaseTransformer):
            The transformer to evaluate.
        dataset_generator (rdt.tests.datasets.BaseDatasetGenerator):
            The dataset generator to performance tests against.
        should_assert (bool):
            Whether or not to raise AssertionErrors.
        results (dict or None):
            If not None, performance results will be collected and stored in results.
    """
    transformer_instance = transformer()
    expected = dataset_generator.get_performance_thresholds()

    dataset_sizes = _get_dataset_sizes(dataset_generator.DATA_TYPE)
    for sizes in dataset_sizes:
        fit_size, transform_size = sizes

        out = profile_transformer(
            transformer=transformer_instance,
            dataset_generator=dataset_generator,
            fit_size=fit_size,
            transform_size=transform_size,
        )

        for test_name in TEST_NAMES:
            metric = test_name.split(' ', maxsplit=-1)[-1].lower()
            function_name = '_'.join(test_name.split(' ')[:-1]).lower()
            size = fit_size if function_name == 'fit' else transform_size

            valid = True
            try:
                _validate_metric_against_threshold(
                    out[test_name], expected[function_name][metric], size)
            except AssertionError as error:
                valid = False
                if should_assert:
                    raise error

            if results is not None:
                max_metric, prev_valid = results.get(test_name, (0, True))
                max_metric = max(max_metric, out[test_name] / size)
                results[test_name] = (max_metric, prev_valid and valid)


@pytest.mark.parametrize(('transformer', 'dataset_generator'), test_cases)
def test_performance(transformer, dataset_generator):
    """Run the performance tests for RDT.

    This test should find all relevant transformers for the given
    dataset generator, and run the ``profile_transformer``
    method, which will assert that the memory consumption
    and times are under the maximum acceptable values.

    Input:
        transformer (rdt.transformers.BaseTransformer):
            The transformer to test.
        dataset_generator (rdt.tests.dataset.BaseDatasetGenerator):
            The dataset generator to performance tests against.
    """
    validate_performance(transformer, dataset_generator, should_assert=True)


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
