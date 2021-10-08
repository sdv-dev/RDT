"""Test whether the performance of the Transformers is the expected one."""

import importlib

import numpy as np
import pandas as pd
import pytest

from rdt.transformers import get_transformers_by_type
from tests.datasets import BaseDatasetGenerator, get_dataset_generators_by_type
from tests.performance.profiling import profile_transformer

TEST_NAMES = [
    'Fit Time',
    'Transform Time',
    'Reverse Transform Time',
    'Fit Memory',
    'Transform Memory',
    'Reverse Transform Memory',
]


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
dataset_generators = BaseDatasetGenerator.get_subclasses()
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


def validate_performance(dataset_generator, should_assert=False, desired_transformer=None):
    """Validate the performance of all transformers for a dataset_generator.

    Args:
        dataset_generator (rdt.tests.datasets.BaseDatasetGenerator):
            The dataset generator to performance tests against.
        should_assert (bool):
            Whether or not to raise AssertionErrors.
        desired_transformer (rdt.transformers.BaseTransformer or None):
            The transformer to record performance for, if specified.

    Output:
        pandas.DataFrame:
            The performance results for all transformers evaluated.
    """
    transformers = transformer_map.get(dataset_generator.DATA_TYPE, [])

    expected = dataset_generator.get_performance_thresholds()

    dataset_sizes = _get_dataset_sizes(dataset_generator.DATA_TYPE)

    results = pd.DataFrame(
        [[test_name, None, 0, 0, True] for test_name in TEST_NAMES],
        columns=['Test', 'Value', 'Total', 'NumTransformers', 'Valid'],
    )
    results.loc[:, 'NumTransformers'] = len(transformers)

    for transformer in transformers:
        transformer_instance = transformer()

        metrics = {}

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

                metrics[test_name] = max(metrics.get(test_name, 0), out[test_name] / size)

                try:
                    _validate_metric_against_threshold(
                        out[test_name], expected[function_name][metric], size)
                except AssertionError as error:
                    results.loc[results.Test == test_name, 'Valid'] = False

                    if should_assert:
                        raise error

        for test_name in TEST_NAMES:
            results.loc[results.Test == test_name, 'Total'] += metrics[test_name]
            if desired_transformer and desired_transformer == transformer:
                results.loc[results.Test == test_name, 'Value'] = metrics[test_name]

    return results


def validate_performance_for_transformer(desired_transformer):
    """Validate performance for the desired transformer.

    Validate the performance of the specified transformer against all
    transformers of that data type.

    Args:
        desired_transformer (rdt.transformers.BaseTransformer):
            The transformer to evaluate.

    Output:
        pandas.DataFrame:
            The average performance results for the desired transformer
            against all transformers of that data type.
    """
    dataset_generator_map = get_dataset_generators_by_type()
    data_type = desired_transformer.get_input_type()

    dataset_generators_for_type = dataset_generator_map.get(data_type, [])
    results = pd.DataFrame(
        [[test_name, 0, 0, 0, True] for test_name in TEST_NAMES],
        columns=['Test', 'Value', 'Total', 'NumTransformers', 'Valid'],
    )

    for dataset_generator in dataset_generators_for_type:
        results_for_type = validate_performance(
            dataset_generator,
            desired_transformer=desired_transformer,
        )

        for test_name in TEST_NAMES:
            results.loc[
                results.Test == test_name,
                'Total',
            ] += results_for_type.loc[
                results_for_type.Test == test_name,
                'Total',
            ]
            results.loc[
                results.Test == test_name,
                'NumTransformers',
            ] += results_for_type.loc[
                results_for_type.Test == test_name,
                'NumTransformers',
            ]
            results.loc[
                results.Test == test_name,
                'Valid',
            ] &= results_for_type.loc[
                results_for_type.Test == test_name,
                'Valid',
            ]

        results['Value'] = results['Value'] + results_for_type['Value']

    average_results = pd.DataFrame()
    average_results['Test'] = results['Test']
    average_results['Value'] = results['Value'] / len(dataset_generators_for_type)
    average_results['Average'] = results['Total'] / results['NumTransformers']
    average_results['Valid'] = results['Valid']

    return average_results


@pytest.mark.parametrize('dataset_generator', dataset_generators)
def test_performance(dataset_generator):
    """Run the performance tests for RDT.

    This test should find all relevant transformers for the given
    dataset generator, and run the ``profile_transformer``
    method, which will assert that the memory consumption
    and times are under the maximum acceptable values.

    Input:
        dataset_generator (rdt.tests.dataset.BaseDatasetGenerator):
            The dataset generator to performance tests against.
    """
    validate_performance(dataset_generator, should_assert=True)


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
