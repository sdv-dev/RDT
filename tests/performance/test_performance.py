"""Test whether the performance of the Transformers is the expected one."""

import importlib

import numpy as np
import pandas as pd
import pytest

from rdt.transformers import get_transformers_by_type
from rdt.transformers.numerical import BayesGMMTransformer
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
    sandbox = [BayesGMMTransformer]

    dataset_generators = get_dataset_generators_by_type()
    transformers = get_transformers_by_type()

    for data_type, transformers_for_type in transformers.items():
        dataset_generators_for_type = dataset_generators.get(data_type, [])

        for transformer in transformers_for_type:
            if transformer in sandbox:
                continue

            for dataset_generator in dataset_generators_for_type:
                all_test_cases.append((transformer, dataset_generator))

    return all_test_cases


test_cases = _get_performance_test_cases()


def evaluate_transformer_performance(transformer, dataset_generator):
    """Evaluate the given transformer's performance against the given dataset generator.

    Args:
        transformer (rdt.transformers.BaseTransformer):
            The transformer to evaluate.
        dataset_generator (rdt.tests.datasets.BaseDatasetGenerator):
            The dataset generator to performance test against.

    Returns:
        pandas.DataFrame:
            The performance test results.
    """
    transformer_instance = transformer()

    sizes = _get_dataset_sizes(dataset_generator.DATA_TYPE)

    out = []
    for fit_size, transform_size in sizes:
        performance = profile_transformer(
            transformer=transformer_instance,
            dataset_generator=dataset_generator,
            fit_size=fit_size,
            transform_size=transform_size,
        )
        size = np.array([fit_size, transform_size, transform_size] * 2)
        out.append(performance / size)

    return pd.DataFrame(out).max(axis=0)


def validate_performance(performance, dataset_generator, should_assert=False):
    """Validate the performance of all transformers for a dataset_generator.

    Args:
        performance (pd.DataFrame):
            The performance metrics of a transformer against a dataset_generator.
        dataset_generator (rdt.tests.datasets.BaseDatasetGenerator):
            The dataset generator to performance test against.
        should_assert (bool):
            Whether or not to raise AssertionErrors.

    Returns:
        list[bool]:
            A list of if each performance metric was valid or not.
    """
    expected = dataset_generator.get_performance_thresholds()
    out = []
    for test_name, value in performance.items():
        function, metric = test_name.lower().replace(' ', '_').rsplit('_', 1)
        expected_metric = expected[function][metric]
        valid = value < expected_metric
        out.append(valid)

        if should_assert and not valid:
            raise AssertionError(f'{function} {metric}: {value} > {expected_metric}')

    return out


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
    performance = evaluate_transformer_performance(transformer, dataset_generator)
    validate_performance(performance, dataset_generator, should_assert=True)


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
