import importlib
import re

import pytest

from rdt.hyper_transformer import HyperTransformer
from rdt.transformers import (
    AnonymizedFaker,
    BaseTransformer,
    BinaryEncoder,
    ClusterBasedNormalizer,
    FloatFormatter,
    GaussianNormalizer,
    RegexGenerator,
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
        *[
            (class_, '__init__', 'enforce_uniqueness')
            for class_ in (
                RegexGenerator,
                AnonymizedFaker,
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
    'class_, method, parameter, value, error',
    [
        *[
            (
                GaussianNormalizer,
                '__init__',
                'distribution',
                value,
                KeyError,
            )
            for value in ('gaussian', 'student_t', 'truncated_gaussian')
        ],
    ],
)
def test_deprecated_parameters_with_value(class_, method, parameter, value, error):
    """Test that deprecated parameters raise an error."""
    # Setup
    instance = class_()

    # Run and Assert
    with pytest.raises(error):
        getattr(instance, method)(**{parameter: value})


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
        'rdt.transformers.categorical.CustomLabelEncoder',
        'rdt.transformers.id.IDGenerator',
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


def test_text_sdtype():
    """Test that the text sdtype is no longer supported."""
    # Setup
    ht = HyperTransformer()
    ht.field_sdtypes = {'col_text': 'text'}

    # Run
    supported_sdtypes = ht._get_supported_sdtypes()

    # Assert
    assert 'text' not in supported_sdtypes
