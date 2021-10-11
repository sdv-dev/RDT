"""RDT code style module."""

import importlib
import inspect
import json
import re
from pathlib import Path
from types import FunctionType

from rdt.transformers.base import BaseTransformer


def validate_transformer_name(transformer):
    """Return whether or not the ``Transformer`` ends with the ``Transformer`` nomenclature."""
    return transformer.__name__.endswith('Transformer')


def validate_transformer_subclass(transformer):
    """Return whether or not the ``Transformer`` is a subclass of ``BaseTransformer``."""
    return issubclass(transformer, BaseTransformer)


def validate_transformer_module(transformer):
    """Return whether or not the ``Transformer`` is inside the right module."""
    transformer_file = Path(inspect.getfile(transformer))
    transformer_folder = transformer_file.parent

    is_valid = False
    if transformer_folder.match('transformers'):
        is_valid = True

    elif transformer_folder.parent.match('transformers'):
        is_valid = True

    elif transformer_folder.parent.match('addons'):
        is_valid = True

    return is_valid


def _validate_config_json(config_path, transformer_name):

    with open(config_path, 'r', encode='utf-8') as config_file:
        config_dict = json.load(config_file)

    required_keys = ['name', 'transformers']
    has_required_keys = set(config_dict.keys()).issuperset(required_keys)
    config_is_valid = [has_required_keys]

    declared = False
    transformers = config_dict.get('transformers', [])
    for transformer in transformers:
        package, name = transformer.rsplit('.', 1)
        if name == transformer_name:
            declared = True
            try:
                getattr(importlib.import_module(package), name)
                imported = True
            except ImportError:
                imported = False

            config_is_valid.extend([declared, imported])

    if len(config_is_valid) > 1:
        return all(config_is_valid)

    return False


def validate_transformer_addon(transformer):
    """Validate if a ``Transformer`` is a valid ``addon``."""
    transformer_file = Path(inspect.getfile(transformer))
    transformer_folder = transformer_file.parent

    is_addon = transformer_folder.parent.match('addons')
    if is_addon:
        documents = list(transformer_folder.iterdir())
        for document in documents:
            if document.match('__init__.py'):
                init_file_exist = True

            elif document.match('config.json'):
                config_json_exist = True
                config_json_is_valid = _validate_config_json(document, transformer.__name__)

            elif re.match(r'^[a-z]+.*.py$', document.name):
                module_py = True

        return all([init_file_exist, config_json_exist, config_json_is_valid, module_py])


def _get_test_location(transformer):
    transformer_file = Path(inspect.getfile(transformer))
    transformer_folder = transformer_file.parent
    rdt_unit_test_path = Path(__file__).parent / 'unit'
    test_location = None

    if transformer_folder.match('transformers'):
        test_location = rdt_unit_test_path / 'transformers' / f'test_{transformer_file.name}'

    elif transformer_folder.parent.match('transformers'):
        test_location = rdt_unit_test_path / 'transformers' / transformer_folder
        test_location = test_location / f'test_{transformer_file.name}'

    elif transformer_folder.parent.match('addons'):
        test_location = rdt_unit_test_path / 'transformers' / 'addons'
        test_location = test_location / f'test_{transformer_file.name}'

    return test_location


def validate_test_location(transformer):
    """Validate if the test file exists in the expected location."""
    test_location = _get_test_location(transformer)
    if test_location is None:
        return False

    return test_location.exists()


def validate_test_names(transformer):
    """Validate if the test methods are properly specified."""
    test_file = _get_test_location(transformer)

    # load the module
    spec = importlib.util.spec_from_file_location(f'Test{transformer.__name__}', test_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # get test class
    try:
        test_class = getattr(module, f'Test{transformer.__name__}')

    except AttributeError:
        return False  # The test class doesn't exist

    # get the test methods that start with test
    test_functions = inspect.getmembers(test_class, predicate=inspect.isfunction)
    test_functions = [
        test
        for test, _ in test_functions
        if test.startswith('test')
    ]

    if not test_functions:
        return False  # No test functions found

    transformer_functions = [
        name
        for name, function in transformer.__dict__.items()
        if isinstance(function, (FunctionType, classmethod, staticmethod))
    ]

    test_function_exists = []

    for transformer_function in transformer_functions:
        count = len(test_function_exists)
        simple_test = f'test_{transformer_function}'
        described_test = f'test_{transformer_function}_'
        for test in test_functions:
            if test.startswith(described_test):
                test_function_exists.append(True)
            if test.startswith(simple_test):
                test_function_exists.append(True)

        if count == len(test_function_exists):
            test_function_exists.append(False)

    return len(test_function_exists) >= len(transformer_functions)
