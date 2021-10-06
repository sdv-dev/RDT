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

    print(pd.DataFrame(results.items(), columns=['Check', 'Correct']))

    valid = True
    for check_result in results.values():
        valid = valid and check_result

    return valid
