import importlib
import re

import pytest

from rdt.hyper_transformer import HyperTransformer
from rdt.transformers import (
    BaseTransformer,
    BinaryEncoder,
    ClusterBasedNormalizer,
    FloatFormatter,
    GaussianNormalizer,
    UnixTimestampEncoder,
)


@pytest.mark.parametrize(
    'class_, method, parameter',
    [
        (
            HyperTransformer,
            'update_transformers_by_sdtype',
            'transformer',
        ),
        *[
            (class_, '__init__', 'model_missing_values')
            for class_ in (
                FloatFormatter,
                BinaryEncoder,
                UnixTimestampEncoder,
                GaussianNormalizer,
                ClusterBasedNormalizer,
            )
        ],
    ],
)
def test_deprecated_parameters(class_, method, parameter):
    """Test that deprecated parameters raise an error."""
    # Setup
    instance = class_()
    expected_message = (
        f"{class_.__name__}.{method}() got an unexpected keyword argument '{parameter}'"
    )

    # Run and Assert
    with pytest.raises(TypeError, match=re.escape(expected_message)):
        getattr(instance, method)(**{parameter: 'value'})


@pytest.mark.parametrize(
    'class_, method, parameter',
    [
        (BaseTransformer, 'get_input_sdtype', None),
    ],
)
def test_deprecated_methods(class_, method, parameter):
    """Test that deprecated methods raise an error."""
    # Setup
    instance = class_()
    expected_message = f"'{class_.__name__}' object has no attribute '{method}'"

    # Run and Assert
    with pytest.raises(AttributeError, match=re.escape(expected_message)):
        getattr(instance, method)(**{parameter: 'value'} if parameter else {})


@pytest.mark.parametrize(
    'class_path',
    [
        'rdt.transformers.FrequencyEncoder',
        'rdt.transformers.categorical.FrequencyEncoder',
    ],
)
def test_deprecated_classes(class_path):
    """Test that deprecated classes can no longer be imported."""
    # Setup
    module_path, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    expected_message = f"module '{module_path}' has no attribute '{class_name}'"

    # Run and Assert
    with pytest.raises(AttributeError, match=re.escape(expected_message)):
        getattr(module, class_name)
