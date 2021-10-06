import pandas as pd

from tests.integration.test_transformers import validate_transformer


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
    results = validate_transformer(transformer)

    check_descriptions = [
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
    results_summary = pd.DataFrame(results.items(), columns=['Check', 'Correct'])
    results_summary['Details'] = check_descriptions
    print(results_summary)

    valid = True
    for check_result in results.values():
        valid = valid and check_result

    return valid
