import importlib

import pandas as pd
from tabulate import tabulate

from tests.integration.test_transformers import validate_transformer

CHECK_DETAILS = [
    'At least one Dataset Generator exists for the Transformer data type',
    'The Transformer can transform data and produce output(s) of the indicated data type(s)',
    ('The Transformer can reverse transform the data it produces, '
     'going back to the original data type'),
    ('Transforming data and reversing it recovers the original data, '
     'if composition is identity is specified'),
    'The HyperTransformer is able to use the Transformer and produce float values',
    ('The HyperTransformer is able to reverse the data that it has previously transformed '
     'and restore the original data type'),
]


def get_class(class_name):
    """Get the specified class.

    Args:
        class (str):
            Full name of class to import.
    """
    obj = None
    if isinstance(class_name, str):
        package, name = class_name.rsplit('.', 1)
        obj = getattr(importlib.import_module(package), name)

    return obj


def validate_transformer_integration(transformer):
    """Validate the integration tests of a transformer.

    This function runs the automated integration test functions on the Transformer.
    It will print to console a summary of the integration tests, along with which
    checks have passed or failed.

    Args:
        transformer (string or rdt.transformers.BaseTransformer):
            The transformer to validate.
    Output:
        bool:
            Whether or not the transformer passes all integration checks.
    """
    print(f'Validating Integration Tests for transformer {transformer}\n')

    if isinstance(transformer, str):
        transformer = get_class(transformer)

    results = validate_transformer(transformer)

    valid = True
    for check_result in results.values():
        valid = valid and check_result

    if valid:
        print('SUCCESS: The integration tests were successful.\n')
    else:
        print('ERROR: One or more integration tests were not successful.\n')

    results_summary = pd.DataFrame(results.items(), columns=['Check', 'Correct'])
    results_summary['Details'] = CHECK_DETAILS
    print(tabulate(results_summary, headers='keys', showindex=False))

    return valid
