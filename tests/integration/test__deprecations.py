import re

import pytest

from rdt.hyper_transformer import HyperTransformer


@pytest.mark.parametrize(
    'method, parameter, expected_message',
    [
        (
            'update_transformers_by_sdtype',
            'transformer',
            'HyperTransformer.update_transformers_by_sdtype() got an unexpected keyword'
            " argument 'transformer'. Did you mean 'transformer_name'?",
        ),
    ],
)
def test_deprecated_parameters(method, parameter, expected_message):
    """Test that deprecated parameters raise an error."""
    # Setup
    ht = HyperTransformer()

    # Run and Assert
    with pytest.raises(TypeError, match=re.escape(expected_message)):
        getattr(ht, method)(**{parameter: 'value'})
