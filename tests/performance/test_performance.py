"""Test whether the performance of the Transformers is the expected one."""

import numpy as np
import pandas as pd
import pytest

from rdt.performance.datasets import get_dataset_generators_by_type
from rdt.performance.performance import evaluate_transformer_performance
from rdt.performance.profiling import profile_transformer
from rdt.transformers import get_transformers_by_type
from rdt.transformers.categorical import CustomLabelEncoder
from rdt.transformers.numerical import ClusterBasedNormalizer

SANDBOX_TRANSFORMERS = [ClusterBasedNormalizer, CustomLabelEncoder]


def _get_performance_test_cases():
    """Get all the (transformer, dataset_generator) combinations for testing."""
    all_test_cases = []

    dataset_generators = get_dataset_generators_by_type()
    transformers = get_transformers_by_type()

    for sdtype, transformers_for_type in transformers.items():
        dataset_generators_for_type = dataset_generators.get(sdtype, [])

        for transformer in transformers_for_type:
            if transformer in SANDBOX_TRANSFORMERS:
                continue

            for dataset_generator in dataset_generators_for_type:
                all_test_cases.append((transformer, dataset_generator))

    return all_test_cases


test_cases = _get_performance_test_cases()


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
