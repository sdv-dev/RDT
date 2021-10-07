import importlib

import pandas as pd
from tabulate import tabulate

from tests.integration.test_transformers import VALIDATION_PASS, validate_transformer

CHECK_DETAILS = {
    '_validate_dataset_generators': (
        'At least one Dataset Generator exists for the Transformer data type.',
    ),
    '_validate_transformed_data': (
        'The Transformer can transform data and produce output(s) of the indicated data type(s).',
    ),
    '_validate_reverse_transformed_data': (
        'The Transformer can reverse transform the data it produces, going back to the ',
        'original data type.',
    ),
    '_validate_composition': (
        'Transforming data and reversing it recovers the original data, if composition is ',
        'identity is specified.',
    ),
    '_validate_hypertransformer_transformed_data': (
        'The HyperTransformer is able to use the Transformer and produce float values.',
    ),
    '_validate_hypertransformer_reverse_transformed_data': (
        'The HyperTransformer is able to reverse the data that it has previously transformed ',
        'and restore the original data type.',
    ),
}


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
        valid = valid and check_result == VALIDATION_PASS

    if valid:
        print('SUCCESS: The integration tests were successful.\n')
    else:
        print('ERROR: One or more integration tests were NOT successful.\n')

    result_summaries = []
    for test, result in results.items():
        correct = 'Yes' if result == VALIDATION_PASS else 'No'
        detail = CHECK_DETAILS[test] if result == VALIDATION_PASS else result
        result_summaries.append([test, correct, detail])

    summary = pd.DataFrame(result_summaries, columns=['Check', 'Correct', 'Details'])
    print(tabulate(summary, headers='keys', showindex=False))

    return valid
