"""Validation methods for contributing to RDT."""


import importlib
import traceback

import pandas as pd
from tabulate import tabulate

from tests.integration.test_transformers import validate_transformer

# Mapping of validation method to (check name, check description).
CHECK_DETAILS = {
    '_validate_dataset_generators': (
        'Dataset Generators',
        'At least one Dataset Generator exists for the Transformer data type.',
    ),
    '_validate_transformed_data': (
        'Output Types',
        'The Transformer can transform data and produce output(s) of the indicated data type(s).',
    ),
    '_validate_reverse_transformed_data': (
        'Reverse Transform',
        (
            'The Transformer can reverse transform the data it produces, going back to the ',
            'original data type.',
        ),
    ),
    '_validate_composition': (
        'Composition is Identity',
        (
            'Transforming data and reversing it recovers the original data, if composition is ',
            'identity is specified.',
        ),
    ),
    '_validate_hypertransformer_transformed_data': (
        'Hypertransformer can transform',
        'The HyperTransformer is able to use the Transformer and produce float values.',
    ),
    '_validate_hypertransformer_reverse_transformed_data': (
        'Hypertransformer can reverse transform',
        (
            'The HyperTransformer is able to reverse the data that it has previously transformed ',
            'and restore the original data type.',
        ),
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

    steps = []
    validation_error = None
    error_trace = None

    try:
        validate_transformer(transformer, steps=steps)
    except Exception as error:
        error_trace = ''.join(traceback.TracebackException.from_exception(error).format())

        for check in CHECK_DETAILS:
            if check in error_trace:
                validation_error = str(error)

    if validation_error is None and error_trace is None:
        print('SUCCESS: The integration tests were successful.\n')
    elif validation_error:
        print('ERROR: One or more integration tests were NOT successful.\n')
    elif error_trace:
        print('ERROR: Transformer errored out with the following error:\n')
        print(error_trace)

    result_summaries = []
    seen_checks = set()
    failed_step = None if validation_error is None else steps[-1]
    for i in range(len(steps)):
        check, details = CHECK_DETAILS[steps[i]]
        if check in seen_checks:
            continue

        seen_checks.update([check])

        if failed_step and steps[i] == failed_step:
            result_summaries.append([check, 'No', validation_error])
        else:
            result_summaries.append([check, 'Yes', details])

    summary = pd.DataFrame(result_summaries, columns=['Check', 'Correct', 'Details'])
    print(tabulate(summary, headers='keys', showindex=False))

    return validation_error is None and error_trace is None
