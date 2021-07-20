import glob
import importlib
import json
import os

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


BASE = os.path.dirname(__file__)
TESTS = glob.glob(BASE + '/test_cases/*.json')


@pytest.mark.parametrize('config_path', TESTS)
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

    transformer = get_instance(config['transformer'], **config['kwargs'])
    dataset_gen = get_instance(config['dataset'])

    out = profile_transformer(transformer, dataset_gen, config['transform_size'],
                              config['fit_size'])

    assert out['Fit Time'] < config['expected']['fit']['time']
    assert out['Fit Memory'] < config['expected']['fit']['memory']
    assert out['Transform Time'] < config['expected']['transform']['time']
    assert out['Transform Memory'] < config['expected']['transform']['memory']
    assert out['Reverse Transform Time'] < config['expected']['reverse_transform']['time']
    assert out['Reverse Transform Memory'] < config['expected']['reverse_transform']['memory']
