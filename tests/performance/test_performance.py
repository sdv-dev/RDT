"""Test whether the performance of the Transformers is the expected one."""

import hashlib
import importlib
import json
import os
import pathlib

import numpy as np
import pandas as pd
import pytest

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


TEST_CASES_PATH = pathlib.Path(__file__).parent / 'test_cases'
TEST_CASES_PATH_LEN = len(str(TEST_CASES_PATH)) + 1
TEST_CASES = [str(test_case) for test_case in TEST_CASES_PATH.rglob('*.json')]
IDS = [test_case[TEST_CASES_PATH_LEN:] for test_case in TEST_CASES]


@pytest.mark.parametrize('config_path', TEST_CASES, ids=IDS)
def test_performance(config_path):
    """Run the performance tests for RDT.

    This test should loop through every test config file,
    load the transformer and dataset generator needed,
    run the ``profile_transformer`` method against them
    and assert that the memory consumption and times are under
    the maximum acceptable values.

    Input:
    - Transformer loaded from config
    - Dataset generator loaded from config
    - fit size loaded from config
    - transform size loaded from config

    Output:
    - pd.Series containing the memory and time for ``fit``,
    ``transform`` and ``reverse_transform``. This should be
    don for each specified test config file.
    """
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    transformer_instance = get_instance(config['transformer'], **config['kwargs'])
    dataset_generator = get_instance(config['dataset'])

    out = profile_transformer(
        transformer=transformer_instance,
        dataset_generator=dataset_generator,
        fit_size=config['fit_size'],
        transform_size=config['transform_size'],
    )

    assert out['Fit Time'] < config['expected']['fit']['time']
    assert out['Fit Memory'] < config['expected']['fit']['memory']
    assert out['Transform Time'] < config['expected']['transform']['time']
    assert out['Transform Memory'] < config['expected']['transform']['memory']
    assert out['Reverse Transform Time'] < config['expected']['reverse_transform']['time']
    assert out['Reverse Transform Memory'] < config['expected']['reverse_transform']['memory']


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


def make_test_case_config(transformer_class, transformer_kwargs, dataset_generator,
                          fit_size, transform_size, iterations=1, multiplier=5,
                          output_path=None, config_name=None):
    """Create a Test Case JSON file for the indicated transformer and dataset.

    If output path is not given, the test case is created with the filename
    ``{config_name}_{dataset_generator}_{fit_size}_{transform_size}.json``
    inside the folder ``{transformer_module}/{transformer_class_name}``.

    Args:
        transformer_class (type):
            Class of the transformer to use.
        tranformer_kwargs (dict):
            Keyword arguments to pass to the transformer.
        dataset_generator (type):
            Dataset Generator class.
        fit_size (int):
            Number of values to use when fitting the transformer.
        transform_size (int):
            Number of values to use when transforming and reverse transforming.
        iterations (int):
            Number of iterations to perform.
        multiplier (int):
            The value used to multiply the average results before rounding them
            up/down. Defaults to 5.
        output_path (str):
            Optional. Path where the output JSON file is written.
        config_name (str):
            Name that should be given to this kwargs config. If not given,
            and default args are used, name is ``default``. Otherwise, a
            hash is taken.
    """
    transformer_instance = transformer_class(**transformer_kwargs)
    outputs = find_transformer_boundaries(
        transformer=transformer_instance,
        dataset_generator=dataset_generator,
        fit_size=fit_size,
        transform_size=transform_size,
        iterations=iterations,
        multiplier=multiplier
    )
    test_case = {
        'dataset': get_fqn(dataset_generator),
        'transformer': get_fqn(transformer_class),
        'kwargs': transformer_kwargs,
        'fit_size': fit_size,
        'transform_size': transform_size,
        'expected': {
            'fit': {
                'time': outputs['Fit Time'],
                'memory': outputs['Fit Memory'],
            },
            'transform': {
                'time': outputs['Transform Time'],
                'memory': outputs['Transform Memory'],
            },
            'reverse_transform': {
                'time': outputs['Reverse Transform Time'],
                'memory': outputs['Reverse Transform Memory'],
            },
        }
    }
    if output_path is None:
        if config_name:
            config_str = config_name
        elif transformer_kwargs:
            config_hash = hashlib.md5()
            config_hash.update(json.dumps(sorted(transformer_kwargs)).encode())
            config_str = config_hash.hexdigest()[0:8]
        else:
            config_str = 'default'

        file_name = f'{config_str}_{dataset_generator.__name__}_{fit_size}_{transform_size}.json'
        module_name = transformer_class.__module__.rsplit('.', 1)[1]
        output_path = TEST_CASES_PATH / module_name / transformer_class.__name__ / file_name

    os.makedirs(output_path.parent, exist_ok=True)
    with open(output_path, 'w') as output_file:
        json.dump(test_case, output_file, indent=4)

    cwd = os.getcwd()
    if str(output_path).startswith(cwd):
        output_path = str(output_path)[len(cwd) + 1:]

    print(f'Test case created: {output_path}')


def make_test_case_configs(transformers, dataset_generators, fit_transform_sizes,
                           iterations=1, multiplier=5):
    """Create Test Case JSON files for multiple transformers and dataset generators.

    Args:
        transformers (List[Union[type, tuple[type, kwargs, name]]]):
            List of transformer classes or transformer_class + kwargs tuples.
        dataset_generators (List[type]):
            List of dataset generator classes.
        fit_transform_sizes (List[tuple[int, int]]):
            List of tuples indicating fit size and transform size.
        iterations (int):
            Number of iterations to perform.
        multiplier (int):
            The value used to multiply the average results before rounding them
            up/down. Defaults to 5.
    """
    for transformer in transformers:
        if not isinstance(transformer, tuple):
            transformer_class, transformer_kwargs, config_name = transformer, {}, None
        elif len(transformer) == 2:
            transformer_class, transformer_kwargs = transformer
            config_name = None
        else:
            transformer_class, transformer_kwargs, config_name = transformer

        for dataset_generator in dataset_generators:
            for fit_size, transform_size in fit_transform_sizes:
                make_test_case_config(
                    transformer_class=transformer_class,
                    transformer_kwargs=transformer_kwargs,
                    dataset_generator=dataset_generator,
                    fit_size=fit_size,
                    transform_size=transform_size,
                    iterations=iterations,
                    multiplier=multiplier,
                    config_name=config_name,
                )
