import glob
import json
import importlib
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
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    
    transformer = get_instance(config['transformer'], **config['kwargs'])
    dataset_gen = get_instance(config['dataset'])

    out = profile_transformer(transformer, dataset_gen, config['transform_size'], config['fit_size'])

    assert out['Fit Time'] < config['expected']['fit']['time']
    assert out['Fit Memory'] < config['expected']['fit']['memory']
    assert out['Transform Time'] < config['expected']['transform']['time']
    assert out['Transform Memory'] < config['expected']['transform']['memory']
    assert out['Reverse Transform Time'] < config['expected']['reverse_transform']['time']
    assert out['Reverse Transform Memory'] < config['expected']['reverse_transform']['memory']
